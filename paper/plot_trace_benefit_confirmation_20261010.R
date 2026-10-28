# Rendering exclusively in R from independently audited results.
root <- 'D:/project/tangjie/trace_as_state_repro/paper'
d <- read.csv(file.path(root,'evidence_trace_benefit_confirmation_20261010','plot_arms.csv'))
contrasts <- read.csv(file.path(root,'evidence_trace_benefit_confirmation_20261010','plot_contrasts.csv'))
out <- file.path(root,'figures','trace_benefit_confirmation_20261010');dir.create(out,recursive=TRUE,showWarnings=FALSE)
d <- d[match(c('original','original_long','compact','prefix','append'),d$arm),]
cols <- c('#78828B','#A6AFB5','#2878A0','#D28B26','#699A70')
labels <- c('Original','Orig.16K','Compact','Before','After')
draw <- function(){
 layout(matrix(c(1,2,3,3),nrow=2,byrow=TRUE),heights=c(1,1))
 par(mar=c(4,4,2.5,.6),mgp=c(2.5,.6,0),tcl=-.25,family='sans',las=1,cex.main=.9)
 bp <- barplot(d$f1,names.arg=labels,col=cols,border=NA,ylim=c(0,1.12),ylab='Mean F1',main='a  New-graph accuracy',cex.names=.75)
 text(bp,d$f1+.04,formatC(d$f1,digits=3,format='f'),cex=.8)
 bp <- barplot(d$mean_deployment_tokens/1000,names.arg=labels,col=cols,border=NA,ylim=c(0,max(d$mean_deployment_tokens)/1000*1.2),ylab='Full model tokens / answer (thousands)',main='b  Full model token use',cex.names=.75)
 text(bp,d$mean_deployment_tokens/1000,formatC(d$mean_deployment_tokens/1000,digits=2,format='f'),pos=3,cex=.8)
 par(mar=c(4,8,2.5,1))
 x <- contrasts[seq_len(4),]; y <- seq_len(nrow(x))
 plot(x$difference,y,xlim=range(c(0,x$lower,x$upper)) + c(-.025,.025),ylim=c(.5,4.5),yaxt='n',ylab='',xlab='Paired F1 difference',main='c  Paired graph-level uncertainty',pch=19,col='#2878A0')
 abline(v=0,lty=2,col='#9A9A9A')
 segments(x$lower,y,x$upper,y,col='#2878A0',lwd=2)
 axis(2,at=y,labels=c('Trace vs compact','Before vs after','Trace vs original','Trace vs original16K'),las=1,cex.axis=.68)

}
cairo_pdf(file.path(out,'confirmation.pdf'),width=6.7,height=6.5,pointsize=10);draw();dev.off()
png(file.path(out,'confirmation.png'),width=2010,height=1950,res=300,pointsize=10,type='cairo');draw();dev.off()
writeLines(c('Source: paper/plot_trace_benefit_confirmation_20261010.R','Input: independently audited plot_arms.csv','24 graph units; two seeds averaged within unit; descriptive means and paired95% graph bootstrap intervals.','Both trace arms include complete single-source preparation per answer.','See report for paired uncertainty and secondary-comparison limitations.'),file.path(out,'PROVENANCE.txt'))
