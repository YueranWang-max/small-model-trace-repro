# Additional method schematic. All shapes and labels are deterministic R vectors.
out <- 'D:/project/tangjie/trace_as_state_repro/paper/figures/method_pipeline_r_20261014'
dir.create(out, recursive=TRUE, showWarnings=FALSE)
draw <- function() {
 par(mar=c(0,0,0,0), xaxs='i', yaxs='i', family='sans')
 plot.new(); plot.window(xlim=c(0,100),ylim=c(0,100),asp=NA)
 ink <- '#252525'
 rounded <- function(x,y,w,h,fill,r=1.1,border=ink,lwd=.8) {
  angles <- list(seq(pi,pi/2,length.out=16),seq(pi/2,0,length.out=16),seq(0,-pi/2,length.out=16),seq(-pi/2,-pi,length.out=16))
  centers <- list(c(x+r,y+h-r),c(x+w-r,y+h-r),c(x+w-r,y+r),c(x+r,y+r))
  xx <- yy <- numeric()
  for(i in 1:4) {xx<-c(xx,centers[[i]][1]+r*cos(angles[[i]]));yy<-c(yy,centers[[i]][2]+r*sin(angles[[i]]))}
  polygon(xx,yy,col=fill,border=border,lwd=lwd)
 }
 block <- function(x,y,title,fill,w=28,h=7) {rounded(x,y,w,h,fill);text(x+w/2,y+h/2,title,cex=.94,col=ink)}
 path <- function(x,y,col=ink) {lines(x,y,col=col,lwd=.9);n<-length(x);arrows(x[n-1],y[n-1],x[n],y[n],length=.07,angle=22,col=col,lwd=.9)}
 rounded(7,34,38,45,'#F5F5F5',r=1.8,border='#929292',lwd=.65)
 rounded(60,34,38,45,'#F5F5F5',r=1.8,border='#929292',lwd=.65)
 text(26,75,'Source call',cex=1.05,font=2,col=ink)
 text(89,75,'Fresh reader\ncall',cex=.9,font=2,col=ink)
 block(12,38,expression(C+Q),'#C8D9ED')
 block(12,50,'Small model','#BDD6AE')
 block(12,62,expression(paste('Thinking ',T)),'#E8BDCE')
 block(65,38,expression(T+C+Q),'#C8D9ED')
 block(65,50,'Small model','#BDD6AE')
 block(65,62,'Prediction','#E8BDCE')
 path(c(26,26),c(45,50));path(c(26,26),c(57,62))
 path(c(79,79),c(45,50));path(c(79,79),c(57,62))
 block(12,16,expression(paste('Recode ',f)),'#EBE2A4')
 text(26,10,'Complete input',cex=.9,col=ink);path(c(26,26),c(12,16));path(c(26,26),c(23,38))
 text(29,30,expression(C),cex=.9,col=ink)
 path(c(26,49,49,65),c(29,29,40,40),col='#377F85')
 text(42,25,'Complete context',cex=.7,col='#377F85')
 path(c(40,54,54,65),c(65.5,65.5,43,43));text(48,68,expression(T),cex=.9,col=ink)
 text(79,10,expression(paste('Question ',Q)),cex=.9,col=ink);path(c(79,79),c(12,38))
 block(65,84,expression(paste('Decode ',f^{-1})),'#EBE2A4')
 path(c(79,79),c(69,84));path(c(79,79),c(91,96));text(79,98,'Answer',cex=1.05,col=ink)
 text(26,83,'1,024-token output cap',cex=.75,col='#555555')
 text(26,88,'Same model; separate calls',cex=.85,font=2,col=ink)
 text(50,3,expression(paste('Controls:  ',C+Q,'  (direct)       ',C+T+Q,'  (append)')),cex=.8,col=ink)
}
cairo_pdf(file.path(out,'publication_arrowfix.pdf'),width=7.2,height=5.3,family='Arial')
draw();dev.off()
png(file.path(out,'publication_arrowfix.png'),width=3600,height=2650,res=500,type='cairo')
draw();dev.off()
png(file.path(out,'preview_arrowfix.png'),width=2160,height=1590,res=300,type='cairo')
draw();dev.off()
writeLines(c('backend: R base graphics','role: additional method schematic',
 'source: frozen compact-input protocol; no empirical values invented',
 'no information removed; text trace only; same-question source and reader',
 'style reference: Transformer Figure 1, module grouping and rounded blocks only'),file.path(out,'PROVENANCE.txt'))
