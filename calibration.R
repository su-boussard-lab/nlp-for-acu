inputs <- list(    
               "language_bert_predictions_ANY_180.csv",    
               "fusion_bert_predictions_ANY_180.csv"
               )

labelslist <- list("labels_ANY_30.csv", "labels_ANY_180.csv", "labels_ANY_365.csv")

for (x in inputs) {
  ps <- read.csv(x, header=FALSE)
  for (y in labelslist) {
    ys <- read.csv(y, header=FALSE)
    ys <- ys$V2 != "False"
    destination = gsub(".csv",".pdf", x)
    pdf(file = destination)
    plot <- CalibrationCurves::val.prob.ci.2(ps$V2,ys, dostats = FALSE, xlim = {0:1}, ylim = {0:1})
    dev.off()
  }
}
               