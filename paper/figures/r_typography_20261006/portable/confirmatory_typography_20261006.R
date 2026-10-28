# Prospective official-graph results. Run only after independent complete audit.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else getwd(),winslash='/')
dir <- file.path(root,'paper/evidence_verification_20261006')
if(!dir.exists(dir)) dir <- file.path(root,'inputs/paper/evidence_verification_20261006')
g <- read.csv(file.path(dir,'graph_summary.csv'),stringsAsFactors=FALSE)
a <- read.csv(file.path(dir,'arm_summary.csv'),stringsAsFactors=FALSE)
c <- read.csv(file.path(dir,'contrasts.csv'),stringsAsFactors=FALSE)
stopifnot(nrow(g)==36,nrow(a)==3,nrow(c)==2)
out <- if(length(args)>1) args[2] else file.path(root,'paper/figures')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
helper <- file.path(root,'paper/figures/publication_font_helpers_20261006.R')
if(!file.exists(helper)) helper <- file.path(root,'publication_font_helpers_20261006.R')
source(helper)

blue <- '#3675A9'; orange <- '#CB8551'; grey <- '#8C9399'
draw <- function() {
 par(mfrow=c(1,3),family='Arial',cex=6.98/.final_font_base,las=1,bty='l',mgp=c(2.3,.6,0),mar=c(4.5,4.3,3,1))
 ids <- unique(g$row_index)
 plot(NA,xlim=c(.7,3.3),ylim=c(0,1),xaxt='n',xlab='',ylab='Node-set F1')
 axis(1,at=1:3,labels=c('Direct','Append','Prefix'),cex.axis=.8)
 for(id in ids) {
  z <- g[g$row_index==id,];z <- z[match(c('direct','post','pre'),z$arm),]
  lines(1:3,z$f1,col=adjustcolor(if(z$problem_type[1]=='parents') blue else orange,.55),lwd=1)
  points(1:3,z$f1,pch=if(z$problem_type[1]=='parents') 16 else 17,col=adjustcolor(if(z$problem_type[1]=='parents') blue else orange,.65),cex=.7)
 }
 mtext('A  Paired graph outcomes',3,1,adj=0,font=2)
 legend('bottomleft',c('Parents','BFS'),col=c(blue,orange),pch=c(16,17),bty='n',cex=.7)
 par(mar=c(4.5,6.8,3,1))
 plot(NA,xlim=range(c(0,c$lower,c$upper))*1.15,ylim=c(.5,2.5),yaxt='n',xlab='Prefix F1 difference',ylab='')
 abline(v=0,lty=2,col=grey);axis(2,at=c(2,1),labels=c('vs. append','vs. direct'),tick=FALSE,cex.axis=.8)
 segments(c$lower,c(2,1),c$upper,c(2,1),lwd=3,col=blue);points(c$mean_difference,c(2,1),pch=19,col=blue)
 mtext('B  Graph-level contrasts',3,1,adj=0,font=2)
 mtext('Descriptive stratified bootstrap intervals',1,3.2,cex=.6)
 par(mar=c(5.3,4.5,3,1))
 a <- a[match(c('direct','post','pre'),a$arm),]
 plot(1:3,a$mean_deployment_tokens/1000,pch=19,col=c(grey,orange,blue),xaxt='n',ylim=c(0,max(a$mean_deployment_tokens/1000)*1.2),xlab='',ylab='Tokens per answer (thousands)')
 segments(1:3,0,1:3,a$mean_deployment_tokens/1000,col=c(grey,orange,blue),lwd=2)
 axis(1,at=1:3,labels=c('Direct','Append','Prefix'),cex.axis=.8)
 mtext('C  Preparation-inclusive cost',3,1,adj=0,font=2)
 mtext('Trace arms pay five sources;\ndirect pays only reading',1,3.2,cex=.6)
}
publication_export('official_graph_confirmatory_20261006',draw,out,11,4.1)
