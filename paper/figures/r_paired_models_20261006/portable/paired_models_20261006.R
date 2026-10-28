# Paired local configurations on identical frozen graph questions.
# Run after both independent audits report complete=TRUE; no imputation.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else getwd(), winslash='/')
base <- file.path(root,'paper')
if(!dir.exists(file.path(base,'evidence_paired4b_20261006'))) base <- file.path(root,'inputs/paper')
paths <- file.path(base,c('evidence_verification_20261006','evidence_paired4b_20261006'))
readAudit <- function(p) {
 calls <- read.csv(file.path(p,'audited_calls.csv'))
 g <- read.csv(file.path(p,'graph_summary.csv'))
 a <- read.csv(file.path(p,'arm_summary.csv'))
 ct <- read.csv(file.path(p,'contrasts.csv'))
 stopifnot(nrow(calls)==240,nrow(g)==36,nrow(a)==3,nrow(ct)==2)
 list(calls=calls,graph=g,arm=a,contrast=ct)
}
z <- lapply(paths,readAudit)
stopifnot(identical(sort(unique(z[[1]]$graph$row_index)),sort(unique(z[[2]]$graph$row_index))))
models <- c('Qwen3-1.7B','Qwen3-4B'); colors <- c('#3675A9','#CB8551')
arms <- c('direct','post','pre'); armLabels <- c('Direct','Append','Prefix')
out <- if(length(args)>1) args[2] else file.path(root,'paper/figures')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
helper <- file.path(root,'paper/figures/publication_font_helpers_20261006.R')
if(!file.exists(helper)) helper <- file.path(root,'publication_font_helpers_20261006.R')
source(helper)

draw <- function() {
 par(mfrow=c(2,2),family='Arial',cex=6.98/.final_font_base,las=1,bty='l',mgp=c(2.2,.6,0),mar=c(4.2,4.3,2.8,1))
 for(metric in c('f1','em')) {
  plot(NA,xlim=c(.7,3.3),ylim=c(0,1),xaxt='n',xlab='',ylab=if(metric=='f1') 'Node-set F1' else 'Exact match')
  axis(1,at=1:3,labels=armLabels)
  for(i in 1:2) {a <- z[[i]]$arm[match(arms,z[[i]]$arm$arm),];xx <- 1:3+c(-.055,.055)[i]
   lines(xx,a[[metric]],col=colors[i],lwd=1.7);points(xx,a[[metric]],col=colors[i],pch=c(16,17)[i],cex=1.1)
  }
  abline(h=1/6,col='#B2B6BA',lty=3)
  mtext(if(metric=='f1') 'a  Set overlap' else 'b  Exact answers',3,1,adj=0,font=2)
  if(metric=='f1') legend('bottomleft',models,col=colors,pch=c(16,17),lty=1,bty='n',cex=.8)
  mtext('Means across the same 12 graphs; dotted line: always empty',1,2.9,cex=.65)
 }
 par(mar=c(4.2,6.4,2.8,1))
 bounds <- unlist(lapply(z,function(t)c(t$contrast$lower,t$contrast$upper)))
 plot(NA,xlim=range(c(0,bounds))*1.1,ylim=c(.5,2.5),yaxt='n',xlab='Prefix F1 difference',ylab='')
 abline(v=0,lty=2,col='#92979C');axis(2,at=c(2,1),labels=c('vs. append','vs. direct'),tick=FALSE)
 for(i in 1:2) {ct <- z[[i]]$contrast[match(c('pre-post','pre-direct'),z[[i]]$contrast$contrast),];yy <- c(2,1)+c(.08,-.08)[i]
  segments(ct$lower,yy,ct$upper,yy,col=colors[i],lwd=2);points(ct$mean_difference,yy,col=colors[i],pch=c(16,17)[i])
 }
 mtext('c  Within-model F1 contrasts',3,1,adj=0,font=2)
 mtext('Descriptive intervals: resample graph units within each task',1,2.9,cex=.65)
 par(mar=c(4.2,4.3,2.8,1))
 top <- max(unlist(lapply(z,function(t)t$arm$mean_deployment_tokens)))/1000
 plot(NA,xlim=c(.7,3.3),ylim=c(0,top*1.15),xaxt='n',xlab='',ylab='Tokens per answer (thousands)')
 axis(1,at=1:3,labels=armLabels)
 for(i in 1:2) {a <- z[[i]]$arm[match(arms,z[[i]]$arm$arm),];xx <- 1:3+c(-.055,.055)[i]
  segments(xx,0,xx,a$mean_deployment_tokens/1000,col=colors[i],lwd=2);points(xx,a$mean_deployment_tokens/1000,col=colors[i],pch=c(16,17)[i])
 }
 mtext('d  Preparation-inclusive cost',3,1,adj=0,font=2)
 mtext('Trace arms pay all five sources plus one average reader',1,2.9,cex=.65)
}
publication_export('paired_models_20261006',draw,out,9,7)
