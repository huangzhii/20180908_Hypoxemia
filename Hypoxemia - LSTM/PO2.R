setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/总人群")
library(anytime)
library(tictoc)
library(lubridate)
###########################################################
###            Expand data
###########################################################
nonvent = read.csv("mimic_mimiciii_po2_nonvent_bmi.csv", header = F, stringsAsFactors = F)
vent = read.csv("mimic_mimiciii_po2_vent_bmi.csv", header = F, stringsAsFactors = F)
label = levels(unlist(read.table("label.tsv", sep = "\t")))
vent = vent[,-10] # Remove the time series after vent
colnames(nonvent) = label
colnames(vent) = label
vent_out_time = anytime(vent$OUT_TIME)
vent_out_time = as.character(ceiling_date(vent_out_time, "1 hours"))
vent$OUT_TIME = vent_out_time
vent = vent[(as.numeric(anytime(vent$OUT_TIME)) - as.numeric(anytime(vent$ADMI_TIME)))/3600 > 6, ]

nonvent$`is_vent` = 0
vent$`is_vent` = 1

alldata = rbind(vent, nonvent)


alldata$ADMI_TIME = anytime(alldata$ADMI_TIME)
alldata$OUT_TIME = anytime(alldata$OUT_TIME)
alldata$ADMI_TIME_numeric = as.numeric(alldata$ADMI_TIME)
alldata$OUT_TIME_numeric = as.numeric(alldata$OUT_TIME)

# start expanding data
expand.data = NULL
start_time <- proc.time()

allhours = 0
hourslist = NULL
for (i in 1:dim(alldata)[1]){
  if (i %% 2500 == 0){
    message(i, "  -  ", allhours, "    Elapsed: ", round((proc.time() - start_time)[3], 2), " secs")
  }
  row = alldata[i,]
  time.in = row$ADMI_TIME_numeric
  time.out = row$OUT_TIME_numeric
  hours = (time.out - time.in)/3600 + 1
  hourslist = c(hourslist, hours)
  allhours = allhours + hours
}
alldata$hours = hourslist
# expand
alldata.expanded <- alldata[rep(row.names(alldata), alldata$hours), 1:14]

start_time <- proc.time()
currtime_list = NULL
allhours = 0
for (i in 1:dim(alldata)[1]){
  if (i %% 2500 == 0){
    message(i, "  -  ", allhours, "    Elapsed: ", round((proc.time() - start_time)[3], 2), " secs")
  }
  row = alldata[i,]
  time.in = row$ADMI_TIME_numeric
  time.out = row$OUT_TIME_numeric
  hours = (time.out - time.in)/3600
  allhours = allhours + hours
  
  numerical.times = seq(time.in, time.in + 3600*hours, by=3600)
  currtime_list[[i]] = numerical.times
}
currtime_list2 = unlist(currtime_list)#do.call(c, unlist(currtime_list, recursive=FALSE))
times = as.character(anytime(currtime_list2))

alldata.expanded$CURR_TIME = times

setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据")
# write.csv(alldata.expanded, file = "expanded.all.data.csv", row.names = F)
save(alldata.expanded, file = "expanded.all.data.Rdata")

setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据")
# alldata.expanded = read.csv("expanded.all.data.csv", header = T)
load("expanded.all.data.Rdata")


###########################################################
###             merge 生命体征
###########################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/生命体征")
library(dplyr)
library(anytime)

hy_vit3_1 = read.csv("mimic_mimiciii_hy_vit3_1.csv", header = F, stringsAsFactors = F)
hy_vit3_2 = read.csv("mimic_mimiciii_hy_vit3_2.csv", header = F, stringsAsFactors = F)
label = as.character(unlist(t(read.table("vit_label.tsv.txt", sep = "\t"))))
colnames(hy_vit3_1) = label
colnames(hy_vit3_2) = label

hy_vit3_1$IS_VENT_VITSIGN = 1
hy_vit3_2$IS_VENT_VITSIGN = 0
hy_vit3 = rbind(hy_vit3_1, hy_vit3_2)
hy_vit3$VIT_TIME = as.character(anytime(hy_vit3$VIT_TIME))

alldata.expanded$mergekey = paste0(alldata.expanded$ICUSTAY_ID, "_", alldata.expanded$CURR_TIME)
hy_vit3$mergekey = paste0(hy_vit3$ICUSTAY_ID, "_", hy_vit3$VIT_TIME)
alldata.merged = full_join(alldata.expanded, hy_vit3,
                            by = 'mergekey')
alldata.merged <- alldata.merged[!duplicated(alldata.merged$mergekey),]

###########################################################
###             merge 实验室检查
###########################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/实验室检查")
hy_lab3_1 = read.csv("mimic_mimiciii_hy_lab3_1.csv", header = F, stringsAsFactors = F)
hy_lab3_2 = read.csv("mimic_mimiciii_hy_lab3_2.csv", header = F, stringsAsFactors = F)
label = as.character(unlist(t(read.table("LAB_LABEL.TSV.txt", sep = "\t"))))
colnames(hy_lab3_1) = label
colnames(hy_lab3_2) = label
hy_lab3_1$IS_VENT_LAB = 1
hy_lab3_2$IS_VENT_LAB = 0
hy_lab3 = rbind(hy_lab3_1, hy_lab3_2)
hy_lab3$LAB_TIME = as.character(anytime(hy_lab3$LAB_TIME))
hy_lab3$mergekey = paste0(hy_lab3$ICUSTAY_ID, "_", hy_lab3$LAB_TIME)

alldata.merged = full_join(alldata.merged, hy_lab3,
                            by = 'mergekey')
alldata.merged <- alldata.merged[!duplicated(alldata.merged$mergekey),]

###########################################################
###             merge 入量
###########################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/入量")
hy_input_3_1 = read.csv("mimic_mimiciii_hy_input_3_1.csv", header = F, stringsAsFactors = F)
hy_input_3_2 = read.csv("mimic_mimiciii_hy_input_3_2.csv", header = F, stringsAsFactors = F)
label = as.character(unlist(t(read.table("input_label.tsv.txt", sep = "\t"))))
colnames(hy_input_3_1) = label
colnames(hy_input_3_2) = label
hy_input_3_1$IS_VENT_INPUT = 1
hy_input_3_2$IS_VENT_INPUT = 0
hy_input = rbind(hy_input_3_1, hy_input_3_2)
hy_input$INPUT_CHARTTIME = as.character(anytime(hy_input$INPUT_CHARTTIME))
hy_input$mergekey = paste0(hy_input$ICUSTAY_ID, "_", hy_input$INPUT_CHARTTIME)

alldata.merged = full_join(alldata.merged, hy_input,
                            by = 'mergekey')
alldata.merged <- alldata.merged[!duplicated(alldata.merged$mergekey),]

###########################################################
###             merge 出量
###########################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/出量")
hy_output_3_1 = read.csv("mimic_mimiciii_hy_output3_1.csv", header = F, stringsAsFactors = F)
hy_output_3_2 = read.csv("mimic_mimiciii_hy_output3_2.csv", header = F, stringsAsFactors = F)
label = as.character(unlist(t(read.table("outputlabel.tsv.txt", sep = "\t"))))
colnames(hy_output_3_1) = label
colnames(hy_output_3_2) = label
hy_output_3_1$IS_VENT_OUTPUT = 1
hy_output_3_2$IS_VENT_OUTPUT = 0
hy_output = rbind(hy_output_3_1, hy_output_3_2)
hy_output$OUT_CHARTTIME = as.character(anytime(hy_output$OUT_CHARTTIME))
hy_output$mergekey = paste0(hy_output$ICUSTAY_ID, "_", hy_output$OUT_CHARTTIME)

alldata.merged = full_join(alldata.merged, hy_output,
                            by = 'mergekey')
alldata.merged <- alldata.merged[!duplicated(alldata.merged$mergekey),]

###########################################################
###             merge 氧疗
###########################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/氧疗")
oxygentherapy = read.csv("mimic_mimiciii_oxygentherapy3.csv", header = F, stringsAsFactors = F)
label = as.character(unlist(t(read.table("oxygen_label.tsv.txt", sep = "\t"))))
colnames(oxygentherapy) = label
oxygentherapy$IS_VENT_OXYGEN = 0
oxygentherapy$OXYGEN_CHARTTIME = as.character(anytime(oxygentherapy$OXYGEN_CHARTTIME))
oxygentherapy$mergekey = paste0(oxygentherapy$ICUSTAY_ID, "_", oxygentherapy$OXYGEN_CHARTTIME)

alldata.merged = full_join(alldata.merged, oxygentherapy,
                            by = 'mergekey')
alldata.merged <- alldata.merged[!duplicated(alldata.merged$mergekey),]
alldata.merged$OXYGEN[is.na(alldata.merged$OXYGEN)] = 0


###########################################################
###             merge 动脉血气
###########################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/动脉血气")
hy_abg_3_1 = read.csv("mimic_mimiciii_hy_abg_3_1.csv", header = F, stringsAsFactors = F)
hy_abg_3_2 = read.csv("mimic_mimiciii_hy_abg_3_2.csv", header = F, stringsAsFactors = F)
label = as.character(unlist(t(read.table("ABG_LABLE.TSV.txt", sep = "\t"))))
colnames(hy_abg_3_1) = label
colnames(hy_abg_3_2) = label
hy_abg_3_1$IS_VENT_ABG = 1
hy_abg_3_2$IS_VENT_ABG = 0
hy_abg = rbind(hy_abg_3_1, hy_abg_3_2)
hy_abg$ABG_TIME = as.character(anytime(hy_abg$ABG_TIME))
hy_abg$mergekey = paste0(hy_abg$ICUSTAY_ID, "_", hy_abg$ABG_TIME)

alldata.merged = full_join(alldata.merged, hy_abg,
                            by = 'mergekey')
alldata.merged <- alldata.merged[!duplicated(alldata.merged$mergekey),]
colnames(alldata.merged)

###########################################################
###             Remove redundant data
###########################################################

alldata.merged2 = alldata.merged[, c("ICUSTAY_ID.x","SUBJECT_ID.x","HADM_ID.x",
                                     "AGE","LOS","INTIME","OUTTIME","GENDER","HEIGHT","WEIGHT",
                                     "BMI","is_vent","CURR_TIME",
                                     "DIAS_BP","HR","SYS_BP","MEAN_BP",
                                     "RESPRATE","TEMPERATURE","SPO2",
                                     "PH","CA","HCO3","HEMOGLOBIN",
                                     "WBC","RBC","NEU","HEMATOCRIT","PLT","CRP",
                                     "BICARBONATE","ALT","AST","ALB","TOTALBILIRUBIN","TNT",
                                     "CK","CKMB","CR","UN","AMI","LIP",
                                     "BNP","CL","GLU","K","NA_ION","APTT",
                                     "PT","INR","DD","FIB","INPUT","OUTPUT","OXYGEN","FIO2","PCO2","PO2")]
colnames(alldata.merged2)[1:3] = c("ICUSTAY_ID","SUBJECT_ID","HADM_ID")
colnames(alldata.merged2)
alldata.merged2 = alldata.merged2[!is.na(alldata.merged2$ICUSTAY_ID), ]

###########################################################
###       Sort data based on ICUSTAY_ID and TIME
###########################################################
library(anytime)
index2sort = paste0(alldata.merged2$ICUSTAY_ID, "_", alldata.merged2$CURR_TIME)
order = sort.int(index2sort, index.return = T)$ix
alldata.merged2 = alldata.merged2[order,]

setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/")
save(alldata.merged2, file = "expanded.all.data.merged.Rdata")

###########################################################
###             Fill missing values
###########################################################
load("expanded.all.data.merged.Rdata")
colnames(alldata.merged2)
library(zoo)
alldata.merged2[is.na(alldata.merged2$INPUT), "INPUT"] = 0
alldata.merged2[is.na(alldata.merged2$OUTPUT), "OUTPUT"] = 0
# is.na(alldata.merged2$FIO2)
icustayIDlist = unique(alldata.merged2$ICUSTAY_ID)
columns2impute = colnames(alldata.merged2)[c(14:52,55:58)]

# start_time <- proc.time()
# for (i in 1:length(icustayIDlist)){
#   if (i %% 100 == 0){
#     message(i, "  -  Elapsed: ", round((proc.time() - start_time)[3], 2), " secs")
#   }
#   icustayID = icustayIDlist[i]
#   subsetColumns = alldata.merged2[alldata.merged2$ICUSTAY_ID == icustayID, columns2impute]
#   alldata.merged2[alldata.merged2$ICUSTAY_ID == icustayID, columns2impute] = na.locf(na.locf(zoo(subsetColumns), na.rm = F), fromLast = T)
# 
# }

### Expanding original alldata
newrows1 = data.frame(matrix(-Inf, nrow=(length(icustayIDlist)-1), dim(alldata.merged2)[2]))
newrows2 = data.frame(matrix(-Inf, nrow=(length(icustayIDlist)-1), dim(alldata.merged2)[2]))
colnames(newrows1) = colnames(alldata.merged2)
colnames(newrows2) = colnames(alldata.merged2)
newrows1$ICUSTAY_ID = unique(alldata.merged2$ICUSTAY_ID)[1:length(unique(alldata.merged2$ICUSTAY_ID))-1]
newrows2$ICUSTAY_ID = unique(alldata.merged2$ICUSTAY_ID)[2:length(unique(alldata.merged2$ICUSTAY_ID))]
newrows1$CURR_TIME = "9999-12-31 23:59:59"
newrows2$CURR_TIME = "0001-01-01 00:00:00"
newrows1 = data.frame(newrows1)
newrows2 = data.frame(newrows2)

alldata.merged3 = rbind(alldata.merged2, rbind(newrows1, newrows2))

index2sort = paste0(alldata.merged3$ICUSTAY_ID, "_", alldata.merged3$CURR_TIME)
order = sort.int(index2sort, index.return = T)$ix
alldata.merged3 = alldata.merged3[order,]
front.impute = data.frame(na.locf(zoo(alldata.merged3[, columns2impute]), na.rm = F))
back.impute = data.frame(na.locf(zoo(alldata.merged3[, columns2impute]), na.rm = F, fromLast = T))
total.impute = pmax(front.impute, back.impute, na.rm = TRUE)


alldata.merged3[, columns2impute] = total.impute

alldata.merged3 = alldata.merged3[alldata.merged3$SUBJECT_ID != -Inf,]
alldata.merged3[alldata.merged3 == -Inf] = NA

setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/")
save(alldata.merged3, file = "expanded.all.data.merged.imputed.Rdata")

###########################################################
###             Sliding window add labels
###########################################################
require(zoo)
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/")
load("expanded.all.data.merged.imputed.Rdata")
alldata.imputed = alldata.merged3
colnames(alldata.imputed)

alldata.imputed$HYPOXEMIA_CLASS = NA
alldata.imputed$HYPOXEMIA_CLASS[alldata.imputed$PO2>=80] = "Normal"
alldata.imputed$HYPOXEMIA_CLASS[alldata.imputed$PO2<80] = "Mild"
alldata.imputed$HYPOXEMIA_CLASS[alldata.imputed$PO2<60] = "Moderate"
alldata.imputed$HYPOXEMIA_CLASS[alldata.imputed$PO2<40] = "Severe"


icustayIDending = cumsum(table(alldata.imputed$ICUSTAY_ID))
icustayIDending = icustayIDending[1:(length(icustayIDending)-1)]
values2NA = c(icustayIDending+1,icustayIDending+2,icustayIDending+3,icustayIDending+4,icustayIDending+5,icustayIDending+6)
values2NA = sort(values2NA)


# PCO2
PCO2 = zoo(alldata.imputed$PCO2)
PCO2.average <- rollapply(PCO2, width = 6, by = 1, FUN = mean, na.rm = TRUE, align = "left")
PCO2.average = c(rep(NA, 6), PCO2.average[1:(length(PCO2.average)-1)])
PCO2.average[values2NA] = NA
alldata.imputed$PCO2_prev_6_average = PCO2.average

# FIO2
FIO2 = zoo(alldata.imputed$FIO2)
FIO2.average <- rollapply(FIO2, width = 6, by = 1, FUN = mean, na.rm = TRUE, align = "left")
FIO2.average = c(rep(NA, 6), FIO2.average[1:(length(FIO2.average)-1)])
FIO2.average[values2NA] = NA
alldata.imputed$FIO2_prev_6_average = FIO2.average

# OXYGEN
OXYGEN = zoo(alldata.imputed$OXYGEN)
OXYGEN.max <- rollapply(OXYGEN, width = 6, by = 1, FUN = max, na.rm = TRUE, align = "left")
OXYGEN.max = c(rep(NA, 6), OXYGEN.max[1:(length(OXYGEN.max)-1)])
OXYGEN.max[values2NA] = NA
alldata.imputed$OXYGEN_MAX_OBSERVED = OXYGEN.max

# INPUT SUM
INPUT = zoo(alldata.imputed$INPUT)
INPUT.sum <- rollapply(INPUT, width = 6, by = 1, FUN = sum, na.rm = TRUE, align = "left")
INPUT.sum = c(rep(NA, 6), INPUT.sum[1:(length(INPUT.sum)-1)])
INPUT.sum[values2NA] = NA
alldata.imputed$INPUT_prev_6_sum = INPUT.sum

# OUTPUT SUM
OUTPUT = zoo(alldata.imputed$OUTPUT)
OUTPUT.sum <- rollapply(OUTPUT, width = 6, by = 1, FUN = sum, na.rm = TRUE, align = "left")
OUTPUT.sum = c(rep(NA, 6), OUTPUT.sum[1:(length(OUTPUT.sum)-1)])
OUTPUT.sum[values2NA] = NA
alldata.imputed$OUTPUT_prev_6_sum = OUTPUT.sum
alldata.imputed$FLUID_BALANCE = INPUT.sum - OUTPUT.sum


# 生命体征 & 实验室检查
VIT_n_LAB_names = c("DIAS_BP","HR","SYS_BP","MEAN_BP","RESPRATE","TEMPERATURE","SPO2",
                    "PH","CA","HCO3","HEMOGLOBIN","WBC","RBC","NEU","HEMATOCRIT","PLT",
                    "CRP","BICARBONATE","ALT","AST","ALB","TOTALBILIRUBIN","TNT","CK",
                    "CKMB","CR","UN","AMI","LIP","BNP","CL","GLU","K","NA_ION","APTT",
                    "PT","INR","DD","FIB")
VIT_n_LAB = zoo(alldata.imputed[,VIT_n_LAB_names])
VIT_n_LAB.average <- rollapply(VIT_n_LAB, width = 6, by = 1, FUN = mean, na.rm = TRUE, align = "left") # This step costs up to an hour!
VIT_n_LAB.average = data.frame(VIT_n_LAB.average)
save(VIT_n_LAB.average, file = "VIT_n_LAB.average.Rdata")
load("VIT_n_LAB.average.Rdata")

emptymat = data.frame(matrix(NA, nrow=6,ncol=length(VIT_n_LAB_names)))
colnames(emptymat) = VIT_n_LAB_names
VIT_n_LAB.average = rbind(emptymat, VIT_n_LAB.average[1:(dim(VIT_n_LAB.average)[1]-1),])
VIT_n_LAB.average[values2NA, ] = NA
VIT_n_LAB_newnames = paste0(VIT_n_LAB_names, "_prev_6_average")
alldata.imputed[,VIT_n_LAB_newnames] = VIT_n_LAB.average

setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/")
save(alldata.imputed, file = "expanded.all.data.merged.imputed.calculated.Rdata")


###########################################################
### Drop some columns, make complete table
###########################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/")
load("expanded.all.data.merged.imputed.calculated.Rdata")
colnames(alldata.imputed)

droplist = NULL
droplist = c("HEIGHT","WEIGHT","BMI")
remains = NULL
for (i in 1:dim(alldata.imputed)[2]){
  message(colnames(alldata.imputed)[i], "\t", sum(is.na(alldata.imputed[,i])) )
  if (sum(is.na(alldata.imputed[,i])) > 1050000){
    droplist = c(droplist, colnames(alldata.imputed)[i])
  }
  else {
    remains = c(remains, colnames(alldata.imputed)[i])
  }
}

# droplist = c("HEIGHT", "WEIGHT", "BMI",
#              "BICARBONATE", "DD", "FIB", "BNP", "NEU",
#              "AMI", "LIP", "CRP", "ALB", "TNT")

todrop = which(names(alldata.imputed) %in% droplist)
alldata.imputed.shrinked = alldata.imputed[complete.cases(alldata.imputed[,-todrop]), -todrop]
sum(is.na(alldata.imputed.shrinked)) == 0
length(unique(alldata.imputed.shrinked$ICUSTAY_ID))

### Add new column: hours

currtime = as.numeric(anytime(alldata.imputed.shrinked$CURR_TIME))/3600
icustayIDtable = table(alldata.imputed.shrinked$ICUSTAY_ID)
icustayIDtable_cumsum = cumsum(table(alldata.imputed.shrinked$ICUSTAY_ID))
hours = unlist(sapply(icustayIDtable, function(x) rep(1:x)))
alldata.imputed.shrinked$HOURS = hours


# # verify times
# a = data.frame(cbind(alldata.imputed.shrinked$ICUSTAY_ID, currtime))
# aa <- a[order(a$V1, -a$currtime ), ] #sort by id and reverse of abs(value)
# aa1 = aa[ !duplicated(aa$V1), ]              # take the first row within each id
# aa <- a[order(a$V1, a$currtime ), ] #sort by id and reverse of abs(value)
# aa2 = aa[ !duplicated(aa$V1), ]              # take the first row within each id
# timediff = (aa1$currtime - aa2$currtime) + 1
# which(!timediff == icustayIDtable)
# 
# icustayIDtable[43]
# times = alldata.imputed.shrinked[alldata.imputed.shrinked$ICUSTAY_ID == 201024,]$CURR_TIME
# times = as.numeric(anytime(times))/3600
# times[length(times)] - times[1]
# as.numeric(anytime(c("2150-11-01 00:00:00","2150-11-01 01:00:00")))/3600 # bug exists in R

data$RESPIRATORY_FAILURE[data$PO2 >= 60 & data$PCO2 <= 50] = "Normal"
data$RESPIRATORY_FAILURE[data$PO2 >= 60 & data$PCO2 >  50] = "High_PCO2"
data$RESPIRATORY_FAILURE[data$PO2 <  60 & data$PCO2 <  50] = "Respiratory_Failure_I"
data$RESPIRATORY_FAILURE[data$PO2 <  60 & data$PCO2 >  50] = "Respiratory_Failure_II"


setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/")
write.csv(alldata.imputed.shrinked, file = "expanded.all.data.merged.imputed.calculated.shrinked.csv", row.names = F)




####################################################################
####################################################################
####################################################################
####################################################################
####################################################################
setwd("/home/zhihuan/Documents/Cong_Feng/20180908_Hypoxemia/Hypoxemia - LSTM/PO2data/PO2数据/")

### Find significant features
data = read.csv("expanded.all.data.merged.imputed.calculated.shrinked.csv", header = T, stringsAsFactors = F)

HYPOXEMIA_CLASS = data$HYPOXEMIA_CLASS
data = data.frame(data)

data$HYPOXEMIA_CLASS[data$HYPOXEMIA_CLASS == "Normal"] = 1
data$HYPOXEMIA_CLASS[data$HYPOXEMIA_CLASS == "Mild"] = 2
data$HYPOXEMIA_CLASS[data$HYPOXEMIA_CLASS == "Moderate"] = 3
data$HYPOXEMIA_CLASS[data$HYPOXEMIA_CLASS == "Severe"] = 4
data$GENDER[data$GENDER == "M"] = 1
data$GENDER[data$GENDER == "F"] = 2

data2 = data[data$HYPOXEMIA_CLASS <= 2, ]
data3 = data2[1:10000, c("AGE","GENDER","K","NA.","HOURS","HYPOXEMIA_CLASS")]
data4 = as.numeric(as.matrix(data3))
data5 = matrix(data4, ncol = 6)
res = as.data.frame(cor(data5, method="spearman"))
data5 = data.frame(data5)
colnames(data5) = c("AGE","GENDER","K","NA.","HOURS","HYPOXEMIA_CLASS")

colnames(data)

results <- list()
for(i in colnames(data)[c(-1,-2,-3,-6,-7,-10,-40)]){
  print(i)
  aov_res <- kruskal.test(formula(paste(i, "~ HYPOXEMIA_CLASS")), data = data2)
  # aov_res <- aov(formula(paste(i, "~ HYPOXEMIA_CLASS")), data = data2)
  print(unlist(summary(aov_res)))
}

