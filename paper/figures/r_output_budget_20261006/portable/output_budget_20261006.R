# Closed output-budget diagnostic, R-only publication export.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else getwd(),winslash='/')
base <- file.path(root,'paper')
if(!dir.exists(file.path(base,'evidence_output_budget_control_20261006'))) base <- file.path(root,'inputs/paper')
old <- file.path(base,'evidence_paired4b_20261006'); new <- file.path(base,'evidence_output_budget_control_20261006')
a <- read.csv(file.path(old,'arm_summary.csv')); g <- read.csv(file.path(new,'graph_summary.csv'))
calls <- read.csv(file.path(new,'audited_calls.csv')); oldcalls <- read.csv(file.path(old,'audited_calls.csv'))
stopifnot(nrow(a)==3,nrow(g)==12,nrow(calls)==60,nrow(oldcalls)==240,
 all(tolower(as.character(calls$valid))=='true'),sum(calls$done_reason=='length')==0,
 all(g$f1==1),all(g$em==1))
g <- g[order(match(g$problem_type,c('parents','bfs')),g$row_index),]
labels <- paste0(rep(c('P','B'),each=6),rep(1:6,2))
stopifnot(all(g$problem_type==rep(c('parents','bfs'),each=6)))
values <- c(a$mean_deployment_tokens[a$arm=='direct'],mean(g$deployment_tokens),
 a$mean_deployment_tokens[a$arm=='post'],a$mean_deployment_tokens[a$arm=='pre'])/1000
out <- if(length(args)>1) args[2] else file.path(root,'paper/figures')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
helper <- file.path(root,'paper/figures/publication_font_helpers_20261006.R')
if(!file.exists(helper)) helper <- file.path(root,'publication_font_helpers_20261006.R')
source(helper)
blue <- '#3675A9'; orange <- '#CB8551'; grey <- '#737A80'
draw <- function() {
 par(mfrow=c(1,2),family='Arial',cex=6.98/.final_font_base,las=1,bty='l',mgp=c(2.1,.6,0),mar=c(4.7,4.4,2.7,1.1))
 plot(NA,xlim=c(.6,12.4),ylim=c(0,1.05),xaxt='n',xlab='',ylab='Node-set F1')
 axis(1,at=1:12,labels=labels)
 abline(v=6.5,lty=3,col='#B2B6BA')
 segments(1:12,g$old_f1,1:12,g$f1,col='#B2B6BA',lwd=1.2)
 points(1:12,g$old_f1,pch=16,col=blue,cex=1.0)
 points(1:12,g$f1,pch=17,col=orange,cex=1.05)
 legend('bottomleft',c('Original cap: 8,192','New cap: 28,672'),col=c(blue,orange),pch=c(16,17),bty='n',cex=.8)
 mtext('a  Same-question direct accuracy',3,1,adj=0,font=2)
 mtext('P: parents; B: BFS. Each point averages five reader seeds.',1,3.1,cex=.65)
 plot(NA,xlim=c(.65,4.35),ylim=c(0,60),xaxt='n',xlab='',ylab='Tokens per answer (thousands)')
 axis(1,at=1:4,labels=FALSE)
 mtext(c('Direct','Direct'),1,1.25,at=1:2,cex=.8)
 mtext(c('8,192','28,672'),1,2.2,at=1:2,cex=.8)
 mtext(c('Append','Prefix'),1,1.8,at=3:4,cex=.8)
 cols <- c(blue,orange,grey,grey)
 segments(1:4,0,1:4,values,col=cols,lwd=2)
 points(1:4,values,col=cols,pch=c(16,17,15,15),cex=1.1)
 text(1:4,values+3,labels=sprintf('%.1fk',values),cex=.8)
 mtext('b  Preparation-inclusive cost',3,1,adj=0,font=2)
 mtext('Trace conditions pay all five sources and one reader.',1,3.7,cex=.65)
}
publication_export('output_budget_20261006',draw,out,9,4)
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'))
