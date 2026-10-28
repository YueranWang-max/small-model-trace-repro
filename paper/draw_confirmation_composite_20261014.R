# Audited confirmation only: 24 graph units; two seeds averaged within graph.
src <- 'D:/project/tangjie/trace_as_state_repro/paper/evidence_trace_benefit_confirmation_20261010'
out <- 'D:/project/tangjie/trace_as_state_repro/paper/figures/confirmation_composite_r_20261014'
dir.create(out,recursive=TRUE,showWarnings=FALSE)
g <- read.csv(file.path(src,'graph_means.csv'))
a <- read.csv(file.path(src,'plot_arms.csv'))
ct <- read.csv(file.path(src,'plot_contrasts.csv'))
arms <- c('original','original_long','compact','prefix','append')
labels <- c('Original','Original, longer budget','Compact direct','Trace before','Trace after')
cols <- c('#7C8796','#8473AA','#285176','#007E87','#D87854')
stopifnot(nrow(g)==120,length(unique(g$graph_index))==24)
for(k in arms) stopifnot(abs(mean(g$f1[g$arm==k])-a$f1[a$arm==k])<1e-12)
key <- g[g$arm=='compact',];key <- key[order(key$problem_type,key$graph_index),]
m <- sapply(arms,function(k) g$f1[match(paste(key$graph_index,k),paste(g$graph_index,g$arm))])
delta <- m[,4]-m[,3]
stopifnot(all(is.finite(m)),all(m>=0 & m<=1))
primary <- ct[ct$contrast=='prefix-compact',]
stopifnot(abs(mean(delta)-primary$difference)<1e-12)
pal <- colorRampPalette(c('#F5F7FA','#CEDCE7','#67A7AD','#006D77'))(101)
draw <- function() {
 set.seed(20261014)
 par(family='Arial',oma=c(1.2,0,.5,0),cex=.72)
 layout(matrix(1:4,2,2,byrow=TRUE),widths=c(1.15,1),heights=c(1.35,1))
 # A: bounded descriptive density, all actual graph means and distribution summaries.
 par(mar=c(3.8,8.5,3.5,1));plot(NA,xlim=c(-.02,1.02),ylim=c(.4,5.7),xlab='Graph-mean F1',ylab='',yaxt='n',bty='n')
 axis(2,at=5:1,labels=labels,las=1,tick=FALSE,cex.axis=.85)
 abline(v=c(0,.5,1),col='#E6EAED',lwd=.5)
 for(i in seq_along(arms)) {
  x<-g$f1[g$arm==arms[i]];y<-6-i
  den<-density(c(-x,x,2-x),from=0,to=1,bw=.08,n=256)
  h<-den$y/max(den$y)*.38
  polygon(c(den$x,rev(den$x)),c(y+.09+h,rep(y+.09,length(h))),col=adjustcolor(cols[i],alpha.f=.35),border=NA)
  points(x,y-.12+runif(length(x),-.075,.075),pch=21,bg=adjustcolor(cols[i],alpha.f=.65),col=cols[i],cex=.7)
  q<-quantile(x,c(.25,.5,.75));segments(q[1],y-.34,q[3],y-.34,col=cols[i],lwd=3)
  points(q[2],y-.34,pch=21,bg='white',col=cols[i],cex=.65)
 }
 mtext('a  Performance distributions',3,adj=0,line=1.5,font=2,cex=1)
 mtext('24 independent graphs per arm',3,adj=0,line=.3,cex=.75)
 # B: every graph, stable task-grouped order; no cherry-picked successes.
 par(mar=c(3.8,4.5,3.5,1));plot(NA,xlim=c(.5,24.5),ylim=range(c(delta,0))+c(-.08,.12),xlab='Graph units, grouped by task',ylab='Prefix minus compact F1',xaxt='n',bty='n')
 abline(h=0,col='#49535D',lwd=.8)
 rect(seq_along(delta)-.32,pmin(0,delta),seq_along(delta)+.32,pmax(0,delta),col=ifelse(delta>=0,'#007E87','#D87854'),border=NA)
 abline(v=12.5,col='#AAB3BC',lty=3)
 axis(1,at=c(6.5,18.5),labels=key$problem_type[c(1,13)],tick=FALSE)
 segments(1,primary$difference,24,primary$difference,col='#285176',lty=2,lwd=1)
 mtext('b  Paired gains and losses',3,adj=0,line=1.5,font=2,cex=1)
 mtext(sprintf('Mean +%.3f; 95%% CI [%.3f, %.3f]',primary$difference,primary$lower,primary$upper),3,adj=0,line=.3,cex=.75)
 # C: graph × method matrix; fixed color scale and all outcomes.
 par(mar=c(7,6,3,2));plot(NA,xlim=c(.5,5.5),ylim=c(.5,24.5),xlab='',ylab='',axes=FALSE)
 for(i in 1:24)for(j in 1:5)rect(j-.48,24-i+.52,j+.48,24-i+1.48,col=pal[1+round(m[i,j]*100)],border=NA)
 text(1:5,-1.2,c('Original','Longer','Compact','Before','After'),cex=.8,xpd=NA)
 axis(2,at=c(18.5,6.5),labels=key$problem_type[c(1,13)],las=1,tick=FALSE)
 abline(h=12.5,col='white',lwd=2)
 # Explicit common scale prevents arbitrary per-column normalization.
 for(i in 1:100)rect(.5+(i-1)*.05,-4.4,.5+i*.05,-3.6,col=pal[i],border=NA,xpd=NA)
 text(c(.5,3,5.5),-5.8,c('F1 0','0.5','1'),cex=.8,xpd=NA)
 mtext('c  Graph-by-method outcomes',3,adj=0,line=1,font=2,cex=1)
 # D: full source-plus-reader costs; no monetary or latency inference.
 par(mar=c(4.8,4.5,3,1));plot(NA,xlim=c(3200,9000),ylim=c(.22,.92),xlab='Complete model tokens per answer',ylab='Mean F1',bty='n')
 abline(h=seq(.3,.9,.2),col='#E6EAED',lwd=.5)
 z<-a[match(arms,a$arm),]
 segments(z$mean_deployment_tokens[3],z$f1[3],z$mean_deployment_tokens[4],z$f1[4],col='#007E87',lwd=1.3)
 points(z$mean_deployment_tokens,z$f1,pch=21,bg=cols,col='white',cex=1.65,lwd=.8)
 posx<-z$mean_deployment_tokens+c(130,100,130,180,180)
 posy<-z$f1+c(-.035,.035,-.035,.035,-.035)
 text(posx,posy,c('Original','Longer budget','Compact','Trace before','Trace after'),adj=0,cex=.8,col=cols)
 mtext('d  Accuracy and complete cost',3,adj=0,line=1,font=2,cex=1)
 text(3500,.89,'Trace before: +39.8% full tokens vs compact',adj=0,cex=.76,col='#007E87')
}
cairo_pdf(file.path(out,'publication_v3.pdf'),width=8.5,height=7.7,family='Arial');draw();dev.off()
png(file.path(out,'publication_v3.png'),width=4250,height=3850,res=500,type='cairo');draw();dev.off()
png(file.path(out,'preview_v3.png'),width=2550,height=2310,res=300,type='cairo');draw();dev.off()
for(f in c('graph_means.csv','plot_arms.csv','plot_contrasts.csv'))file.copy(file.path(src,f),file.path(out,f),overwrite=TRUE)
writeLines(c('Descriptive density: reflected at F1 boundaries, bandwidth 0.08; actual graph points always shown.',
 'Dot jitter: fixed display seed; no scientific randomness added.',
 'Intervals: existing frozen primary graph bootstrap only; no new tests.',
 'Heatmap: shared [0,1] scale, frozen graph order within task; two seeds averaged within graph.'),file.path(out,'PROVENANCE.txt'))
stopifnot(file.exists(file.path(out,'publication_v3.png')),file.exists(file.path(out,'publication_v3.pdf')))
cat('EXPORTED_TO', normalizePath(out), '\n')
print(file.info(file.path(out,c('publication_v3.png','publication_v3.pdf')))[,'size',drop=FALSE])
