# 20180908 Zhi Huang
library("dplyr")

setwd("/Users/zhi/Desktop/Cong_Feng/20180908_Hypoxemia")
vitalsign1_3 = read.csv("./data/mimic_mimiciii_hypoxemia_vitalsign1_3.csv", header = T)
vitalsign2_3 = read.csv("./data/mimic_mimiciii_hypoxemia_vitalsign2_3.csv", header = T)

vitalsign1_3.complete = vitalsign1_3[complete.cases(vitalsign1_3), ]
vitalsign1_3.complete = vitalsign1_3.complete[vitalsign1_3.complete$vitalsign_charttime < vitalsign1_3.complete$po260_first_charttime, ]
vitalsign1_3.complete = vitalsign1_3.complete[vitalsign1_3.complete$vitalsign_charttime > 0,]

vitalsign1_3.complete$temp = vitalsign1_3.complete$po260_first_charttime - vitalsign1_3.complete$vitalsign_charttime
vitalsign1_3.complete = data.frame(vitalsign1_3.complete)

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

### t-SNE
library(Rtsne)
library(ggplot2)
justplot <- function(data, title = "No title"){
  
  ggplot(data, aes(x=V1, y=V2, fill=OS.Status)) +
    geom_point(size=3, shape=21, stroke = 0.5) +
    guides(colour=guide_legend(override.aes=list(size=6))) +
    xlab("Dim 1") + ylab("Dim 2") +
    ggtitle(title) +
    scale_fill_discrete(name = "Is Hypoxemia") +
    theme_bw(base_size=14) +
    theme(legend.title = element_text(size = 12),
          legend.text = element_text(size = 10),
          legend.key.height=unit(1.5,"line"),
          legend.key.width=unit(1.5,"line"),
          # legend.position = c(0.12, 0.88),
          legend.background = element_rect(color = "black", 
                                           fill = "white", size = 0.5, linetype = "solid", lw),
          plot.title = element_text(size=16, hjust = 0.5),
          axis.title=element_text(size=14,face="bold"),
          axis.text.x = element_text(size=12, angle = 0, hjust = 1),
          axis.text.y = element_text(size=12)) #+
  # scale_fill_manual(values = c("deeppink1", "gainsboro"))
}

tsne_model = Rtsne(as.matrix(vitalsign.all[,2:8]), check_duplicates=FALSE, pca=T, perplexity=20, theta=0.5, dims=2)
d_tsne = as.data.frame(tsne_model$Y)
print("save to file ...")
data = cbind(d_tsne, as.character(vitalsign.all$isHypoxemia))
colnames(data) = c("V1","V2","OS.Status")
justplot(data, "Patient features v.s. hypoxemia")

