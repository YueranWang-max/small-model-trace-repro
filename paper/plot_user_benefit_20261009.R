# All rendering uses R; values come from independently audited fresh calls.
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) args[1] else 'D:/project/tangjie/trace_as_state_repro/paper'
d <- read.csv(file.path(root,'evidence_user_benefit_20261009','plot_arms.csv'))
out <- file.path(root,'figures','user_benefit_20261009');dir.create(out,recursive=TRUE,showWarnings=FALSE)
pal <- c(original='#6C757D',compact='#2878A0',prefix='#D28B26')
draw <- function(){
 par(mfrow=c(2,2),mar=c(4,4,2.4,1),mgp=c(2.3,.6,0),tcl=-.25,family='sans',las=1)
 for(s in c('capacity','short')){
  x <- d[d$stratum==s,]; x <- x[match(c('original','compact','prefix'),x$arm),]
  label <- c('Original','Compact','Compact + trace')
  bp <- barplot(x$f1,names.arg=label,col=unname(pal[x$arm]),border=NA,ylim=c(0,1.12),ylab='Mean F1',main=paste(s,': observed quality'),cex.names=.8)
  text(bp,x$f1+.045,formatC(x$f1,digits=3,format='f'),cex=.8)
  bp <- barplot(x$mean_deployment_tokens/1000,names.arg=label,col=unname(pal[x$arm]),border=NA,ylim=c(0,max(x$mean_deployment_tokens)/1000*1.2),ylab='Full tokens / answer (thousands)',main=paste(s,': complete model tokens'),cex.names=.8)
  text(bp,x$mean_deployment_tokens/1000,formatC(x$mean_deployment_tokens/1000,digits=2,format='f'),pos=3,cex=.8)
 }
}
cairo_pdf(file.path(out,'user_benefit.pdf'),width=10,height=6.5);draw();dev.off()
png(file.path(out,'user_benefit.png'),width=3000,height=1950,res=300,type='cairo');draw();dev.off()
writeLines(c('R source: paper/plot_user_benefit_20261009.R','Input: independent audit plot_arms.csv','Bars are descriptive graph means, not population-equivalence evidence.','Capacity packages differ in context and output settings. Prefix includes complete one-trace preparation.'),file.path(out,'PROVENANCE.txt'))
