# All empirical graphics are drawn in base R. No model-generated imagery.
# Run from repo root: Rscript paper/figures/redesign_all.R
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else getwd(),winslash='/')
out <- if(length(args)>1) args[2] else file.path(root,'paper/figures/r_typography_20261006')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
helper <- file.path(root,'paper/figures/publication_font_helpers_20261006.R')
if(!file.exists(helper)) helper <- file.path(root,'publication_font_helpers_20261006.R')
source(helper)

blue <- '#3675A9'; orange <- '#CB8551'; grey <- '#8C9399'; green <- '#6C947D'
read <- function(p) {
 source <- file.path(root,p);if(!file.exists(source)) source <- file.path(root,'inputs',p); saved <- file.path(out,'inputs',p)
 dir.create(dirname(saved),recursive=TRUE,showWarnings=FALSE);if(normalizePath(source,winslash='/') != normalizePath(saved,winslash='/',mustWork=FALSE)) file.copy(source,saved,overwrite=TRUE)
 read.csv(saved,stringsAsFactors=FALSE)
}
panel <- function(title) {par(mar=c(4.2,4.4,3,1),mgp=c(2.5,.6,0),tcl=-.2,las=1,bty='l',family='Arial',cex=6.98/.final_font_base); mtext(title,3,1.2,adj=0,font=2)}
savefig <- function(name,draw,w=8,h=4.2) {
 publication_export(name,draw,out,w,h,fraction=if(name=='source_qualification_figure') .92 else 1)
}
lab <- c(source='Source',fresh='Direct',question_first='Question first',re2='Re2',prefix='State before',append='State after',set_mode='Set mode',node_majority='Node majority',direct='Direct',task_cue='Task cue',neutral_prefix='Neutral prefix')
d <- read('recovery_bridge_20261003/hybrid_source_qualification/audited_calls.csv');d <- d[d$phase=='scored_source',]
stopifnot(nrow(d)==16,all(tolower(as.character(d$qualified))=='true'))
ids <- c(20,21,420,421,154,155,504,505)
savefig('source_qualification_figure',function(){
 par(mfrow=c(1,2),mar=c(5,8,3,1),mgp=c(2.3,.6,0),family='Arial',las=1,cex=6.98/.final_font_base,bty='n')
 for(k in c('f1','em')) {
  z <- vapply(ids,function(i) mean(d[d$row_index==i,k])*100,numeric(1)); yy <- 8:1
  plot(NA,xlim=c(0,105),ylim=c(.4,8.6),yaxt='n',xlab=if(k=='f1') 'Node-set F1 (%)' else 'Exact match (%)',ylab='',xaxs='i')
  rect(0,c(6.5,2.5),105,c(8.5,4.5),col='#F2F5F7',border=NA)
  axis(2,at=yy,labels=paste(rep(c('Short parents','Short BFS','Long parents','Long BFS'),each=2),ids,sep=' / '),tick=FALSE,cex.axis=.8)
  segments(0,yy,z,yy,col=adjustcolor(blue,.4),lwd=2);points(z,yy,pch=21,bg=blue,col='white',cex=1.4)
  for(j in seq_along(z)) text(if(z[j]<5) z[j]+3 else z[j]-4,yy[j]+.25,sprintf('%.1f',z[j]),adj=if(z[j]<5) 0 else 1,cex=.72,col='#333333')
  mtext(if(k=='f1') 'A  Source competence by graph' else 'B  Exact answers by graph',3,1.1,adj=0,font=2)
 }
},8.4,5.0)
b <- read('recovery_bridge_20261003/hybrid_five_trace_replication/figure_f1_identification_bounds.csv')
i <- read('recovery_bridge_20261003/hybrid_five_trace_replication/figure_input_feasibility.csv')
c <- read('recovery_bridge_20261003/hybrid_five_trace_replication/figure_slot_coverage.csv')
savefig('interrupted_study_figure',function(){
 layout(matrix(1:3,1),widths=c(1.3,1.2,.8));par(family='Arial',cex=6.98/.final_font_base,las=1,bty='l',mgp=c(2.4,.6,0))
 par(mar=c(6,6.5,3,1));y<-rev(seq_len(nrow(b)))
 plot(NA,xlim=c(0,1),ylim=c(.5,length(y)+.5),yaxt='n',xlab='All-plan F1 bounds',ylab='');axis(2,at=y,labels=lab[b$arm],tick=FALSE,cex.axis=.75)
 segments(b$lower,y,b$upper,y,lwd=7,col='#DCE6ED');points(b$lower,y,pch=3,col=blue);points(b$upper,y,pch=3,col=blue)
 mtext('a  Quality remains undetermined',3,1,adj=0,font=2);mtext('Ranges are not confidence intervals',1,4.1,cex=.65)
 par(mar=c(6,6,3,1));y<-5:1
 plot(NA,xlim=c(0,170),ylim=c(.5,5.5),yaxt='n',xlab='Input tokens (thousands)',ylab='');axis(2,at=y,labels=lab[i$arm],tick=FALSE,cex.axis=.7)
 rect(131.072,.5,170,5.5,col='#F8EAE0',border=NA);abline(v=131.072,col=orange,lwd=1.2);abline(v=98.24,lty=2,col=grey)
 points(i$input_tokens/1000,y,pch=ifelse(i$input_tokens>131072,17,19),col=ifelse(i$input_tokens>131072,orange,blue),cex=1.3)
 text(i$input_tokens/1000,y+.27,sprintf('%.1f',i$input_tokens/1000),cex=.65)
 mtext('b  Configured context overflow',3,1,adj=0,font=2);mtext('Row 506; dashed: output reserve\nShaded: beyond 131,072-token context',1,3.8,cex=.65)
 par(mar=c(6,2,3,1));plot.new();plot.window(xlim=c(0,1),ylim=c(0,5))
 mtext('c  Call coverage',3,1,adj=0,font=2)
 for(j in 1:4){yy <- 4.2-(j-1)*.9;rect(0,yy-.14,.12,yy+.14,col=c(blue,orange,grey,'#DDE1E4')[j],border=NA);text(.18,yy,paste(c('Qualified','Budget decline','Unqualified','Not reached')[j],c$slots[j],sep=': '),adj=0,cex=.8)}
 text(0,.45,'480 planned slots\nCalibration excluded',adj=0,cex=.75)
},10,4.5)
d <- read('recovery_bridge_20261003/partial_state_study_20261004/display_arm_summary.csv')
savefig('partial_state_display',function(){
 par(mfrow=c(3,1),family='Arial',cex=6.98/.final_font_base,bty='l',las=1,mgp=c(2.3,.6,0));x<-1:7;cl<-ifelse(d$arm=='prefix',blue,grey)
 par(mar=c(1.5,4.4,2.5,1));plot(x,d$f1_lower,type='n',ylim=c(0,1),xaxt='n',xlab='',ylab='All-plan F1')
 segments(x,d$f1_lower,x,d$f1_upper,col=cl,lwd=5);points(x,d$f1_lower,pch=1,col=cl);points(x,d$f1_upper,pch=1,col=cl);mtext('A  Missing-outcome ranges (not confidence intervals)',3,1,adj=0,font=2)
 par(mar=c(1.5,4.4,2.5,1));plot(x,d$known,type='n',ylim=c(0,34),xaxt='n',xlab='',ylab='Saved / 32');abline(h=32,lty=2,col=grey)
 segments(x,0,x,d$known,lwd=9,col=adjustcolor(cl,.4));text(x,d$known+2,paste0(d$known,'/32'),cex=.8);mtext('B  Coverage differs across arms',3,1,adj=0,font=2)
 par(mar=c(5,4.4,2.5,1));z<-d$planned_prediction_tokens_lower/1000;plot(x,z,type='n',ylim=c(0,max(z)*1.25),xaxt='n',xlab='',ylab='Token lower bound (k)')
 points(x,z,pch=1,col=cl,cex=1.2);arrows(x,z,x,z+max(z)*.09,length=.06,col=orange);axis(1,at=x,labels=lab[d$arm],cex.axis=.68)
 mtext('C  Full source charged; incomplete cost per planned prediction',3,1,adj=0,font=2)
},8.8,7.2)
d <- read('logs/official_graph_indexed_source/interrupted_figure_summary.csv')
savefig('interrupted_indexed_figure',function(){
 par(mfrow=c(2,3),family='Arial',cex=6.98/.final_font_base,bty='l',las=1,mgp=c(2,.5,0))
 for(j in 1:3){ix<-(2*j-1):(2*j);q<-d[ix,];col<-c(blue,orange)
  par(mar=c(3.8,4,3,1));plot(NA,xlim=c(0,1),ylim=c(.5,2.5),yaxt='n',xlab='All-plan F1',ylab='');axis(2,at=2:1,labels=c('On','Off'),tick=FALSE)
  segments(q$f1_lower,2:1,q$f1_upper,2:1,col=col,lwd=4);points(q$f1_lower,2:1,col=col,pch=ifelse(q$known==16,19,1));points(q$f1_upper,2:1,col=col,pch=ifelse(q$known==16,19,1))
  text(.97,2:1,paste0(q$known,'/16'),adj=1,cex=.8);mtext(paste(letters[j],c('Original','Compact','Indexed')[j]),3,1,adj=0,font=2)
 }
 for(j in 1:3){ix<-(2*j-1):(2*j);q<-d[ix,];z<-q$tokens_lower/1000;col<-c(blue,orange)
  par(mar=c(3.8,4,2,1));plot(NA,xlim=c(0,125),ylim=c(.5,2.5),yaxt='n',xlab='Arm token sum (k)',ylab='');axis(2,at=2:1,labels=c('On','Off'),tick=FALSE)
  points(z,2:1,col=col,pch=ifelse(tolower(q$cost_complete)=='true',19,1),cex=1.3);bad<-tolower(q$cost_complete)!='true';arrows(z[bad],(2:1)[bad],z[bad]+10,(2:1)[bad],col=orange,length=.07)
  text(z,2:1+.25,sprintf('%.1f',z),cex=.7);mtext(paste(letters[j+3],'Observed usage'),3,.7,adj=0,font=2)
 }
},9,5.5)
d<-read('paper/figures/data/official_graph_identity_recode.csv');g<-read('paper/figures/data/official_graph_identity_recode_cases.csv')
savefig('official_graph_identity_recode',function(){
 par(mfrow=c(1,3),family='Arial',cex=6.98/.final_font_base,bty='l',las=1,mgp=c(2.3,.6,0));cl<-c(grey,blue,orange)
 par(mar=c(5,4.3,3,1));plot(NA,xlim=c(.7,3.3),ylim=c(0,1),xaxt='n',xlab='',ylab='Graph-mean F1')
 for(k in 1:nrow(g)) lines(1:3,as.numeric(g[k,c('original_f1','compact_f1','long_alias_f1')]),col=adjustcolor(grey,.5),lwd=1)
 points(1:3,d$f1,pch=19,col=cl,cex=1.5);axis(1,at=1:3,labels=c('Original','Compact','Alias'),cex.axis=.75);mtext('A  Paired graph trajectories',3,1,adj=0,font=2)
 par(mar=c(6,4.8,3,1));plot(NA,xlim=c(0,1),ylim=c(.5,3.5),yaxt='n',xlab='Mean F1 / descriptive interval',ylab='');axis(2,at=3:1,labels=c('Original','Compact','Alias'),tick=FALSE,cex.axis=.75)
 segments(d$f1_lo,3:1,d$f1_hi,3:1,col=cl,lwd=3);points(d$f1,3:1,col=cl,pch=19);mtext('B  Uncertainty remains large',3,1,adj=0,font=2)
 par(mar=c(6,4.8,3,1));plot(NA,xlim=c(0,130),ylim=c(.5,3.5),yaxt='n',xlab='Input + output tokens (k)',ylab='');axis(2,at=3:1,labels=c('Original','Compact','Alias'),tick=FALSE,cex.axis=.75)
 segments(d$input_tokens/1000,3:1,(d$input_tokens+d$output_tokens)/1000,3:1,col=cl,lwd=8);segments(0,3:1,d$input_tokens/1000,3:1,col=adjustcolor(cl,.35),lwd=8)
 mtext('C  Cost composition',3,1,adj=0,font=2);mtext('Pale: input; solid: output\n16 saved calls per arm',1,4.5,cex=.7)
},10,4.4)
d<-read('paper/figures/data/official_graph_reasoning_mode.csv');g<-read('paper/figures/data/official_graph_reasoning_mode_groups.csv')
savefig('official_graph_reasoning_mode',function(){
 par(mfrow=c(1,2),family='Arial',cex=6.98/.final_font_base,bty='l',las=1,mgp=c(2.5,.9,0))
 par(mar=c(5,4.5,3,1));plot(NA,xlim=c(.7,4.3),ylim=c(0,1),xaxt='n',xlab='',ylab='Node-set F1')
 for(j in 1:4){typ<-c('parents','parents','bfs','bfs')[j];lo<-c(2000,7000,2000,7000)[j];a<-g[g$type==typ & g$length_bin==lo,];v<-a$f1[match(c('on','off'),a$arm)];segments(j-.12,v[1],j+.12,v[2],col=grey,lwd=2);points(j+c(-.12,.12),v,pch=c(19,17),col=c(blue,orange),cex=1.3)}
 axis(1,at=1:4,labels=c('Parents\nshort','Parents\nlong','BFS\nshort','BFS\nlong'),cex.axis=.8);legend('topright',c('Thinking on','Thinking off'),pch=c(19,17),col=c(blue,orange),bty='n',cex=.8);mtext('A  Quality loss by task and length',3,1,adj=0,font=2)
 par(mar=c(6,4.5,3,1));x<-d$deployment_tokens/1000;plot(NA,xlim=c(55,118),ylim=c(0,.6),xlab='16-graph deployment tokens (k)',ylab='Mean F1')
 segments(x,d$f1_lo,x,d$f1_hi,col=c(blue,orange,green),lwd=2);points(x,d$f1,pch=c(19,17,15),col=c(blue,orange,green),cex=1.4)
 text(x,d$f1+.04,c('Always on','Always off','Fixed route'),cex=.75);mtext('B  Lower cost accompanies lost quality',3,1,adj=0,font=2);mtext('Route reuses saved subsets; not a new arm\nVertical intervals are descriptive',1,4.5,cex=.72)
},8.4,4.7)
cat('Six revised SVG/PDF/PNG exports generated in R from unchanged inputs.\n')
