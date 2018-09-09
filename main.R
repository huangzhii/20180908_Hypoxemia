# 20180908 Zhi Huang
library("dplyr")

# setwd("/Users/zhi/Desktop/Cong_Feng/20180908_Hypoxemia")
vitalsign1_3 = read.csv("./data/mimic_mimiciii_hypoxemia_vitalsign1_3.csv", header = T)
vitalsign2_3 = read.csv("./data/mimic_mimiciii_hypoxemia_vitalsign2_3.csv", header = T)

vitalsign1_3.complete = vitalsign1_3[complete.cases(vitalsign1_3), ]
vitalsign1_3.complete = vitalsign1_3.complete[vitalsign1_3.complete$vitalsign_charttime < vitalsign1_3.complete$po260_first_charttime, ]
vitalsign1_3.complete = vitalsign1_3.complete[vitalsign1_3.complete$vitalsign_charttime > 0,]

vitalsign1_3.complete$temp = vitalsign1_3.complete$po260_first_charttime - vitalsign1_3.complete$vitalsign_charttime
vitalsign1_3.complete = data.frame(vitalsign1_3.complete)

vitalsign1_3.complete <- vitalsign1_3.complete[order(temp),] 
vitalsign1_3.sorted = arrange(vitalsign1_3.complete, temp)

vitalsign1_3.final = vitalsign1_3.sorted[!duplicated(vitalsign1_3.sorted$icustay_id),]
vitalsign1_3.final = arrange(vitalsign1_3.final, icustay_id)

vitalsign1_3.final = vitalsign1_3.final[, -which(names(vitalsign1_3.final) %in% c("po260_first_charttime","vitalsign_charttime", "intime", "temp"))]


vitalsign2_3.complete = vitalsign2_3[complete.cases(vitalsign2_3), ]
vitalsign2_3.complete = vitalsign2_3.complete[vitalsign2_3.complete$vitalsign_charttime > 0,]
vitalsign2_3.sorted = arrange(vitalsign2_3.complete, vitalsign_charttime)
vitalsign2_3.final = vitalsign2_3.sorted[!duplicated(vitalsign2_3.sorted$icustay_id),]
vitalsign2_3.final = arrange(vitalsign2_3.final, icustay_id)
vitalsign2_3.final = vitalsign2_3.final[, -which(names(vitalsign2_3.final) %in% c("vitalsign_charttime", "intime"))]


#preprocessing finised
vitalsign1_3.final$isHypoxemia = 1
vitalsign2_3.final$isHypoxemia = 0

vitalsign.all = rbind(vitalsign1_3.final, vitalsign2_3.final)

write.csv(vitalsign.all, "./data/vitalsign.all.csv")


