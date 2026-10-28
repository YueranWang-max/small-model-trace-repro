# Polish the existing scientific compositions using R vector output only.
base<-'D:/project/tangjie/trace_as_state_repro/paper'
root<-file.path(base,'figures/polished_three_20261015')
ink<-'#22364A';teal<-'#00848D';orange<-'#C7784E';navy<-'#365F85';grey<-'#7E8996';light<-'#DFE6EC'
canvas<-function(h){par(mar=rep(0,4),family='Arial',fg=ink,cex=1);plot.new();plot.window(c(0,180),c(0,h),xaxs='i',yaxs='i',asp=1)}
tt<-function(x,y,s,size=8.2,colour=ink,...)text(x,y,s,cex=size/10,col=colour,...)
hd<-function(x,y,l,s)tt(x,y,paste(l,s),size=10,font=2,adj=0)
route<-function(x,y,col=ink){lines(x,y,col=col,lwd=.9);n<-length(x);arrows(x[n-1],y[n-1],x[n],y[n],col=col,length=.07,angle=22,lwd=.9)}
roundbox<-function(x,y,w,h,col,border=light){r<-1.5;xx<-yy<-numeric();for(i in 1:4){a<-list(seq(pi,pi/2,length.out=20),seq(pi/2,0,length.out=20),seq(0,-pi/2,length.out=20),seq(-pi/2,-pi,length.out=20))[[i]];p<-list(c(x+r,y+h-r),c(x+w-r,y+h-r),c(x+w-r,y+r),c(x+r,y+r))[[i]];xx<-c(xx,p[1]+r*cos(a));yy<-c(yy,p[2]+r*sin(a))};polygon(xx,yy,col=col,border=border,lwd=.65)}
module<-function(x,y,s,col,w=43){roundbox(x,y,w,9,col);tt(x+w/2,y+4.5,s,size=9)}
pipeline<-function(){canvas(140);hd(5,134,'','Complete input, two model calls, one decoded answer')
 roundbox(8,39,58,66,'#F7F9FB');roundbox(108,39,64,66,'#F7F9FB')
 tt(37,100,'Source call',size=10,font=2);tt(140,100,'Fresh reader call',size=10,font=2)
 module(15,45,expression(C+Q),'#E8EFF5');module(15,61,'Small model','#D6E9E7');module(15,77,expression(paste('Thinking ',T)),'#F2E2D8')
 module(118,45,expression(T+C+Q),'#E8EFF5');module(118,61,'Small model','#D6E9E7');module(118,77,'Prediction','#F2E2D8')
 for(x in c(36.5,139.5)){route(c(x,x),c(54,61));route(c(x,x),c(70,77))}
 module(15,18,expression(paste('Recode ',f)),'#F0EEE5');tt(36.5,11,'Complete input',size=9);route(c(36.5,36.5),c(13,18));route(c(36.5,36.5),c(27,45))
 route(c(36.5,79,79,118),c(33,33,47,47),teal);tt(74,27,'Complete context C',size=8,colour=teal)
 route(c(58,91,91,118),c(81.5,81.5,51,51),orange);tt(88,88,'Text trace T',size=8.2,colour=orange)
 tt(139.5,11,expression(paste('Question ',Q)),size=9);route(c(139.5,139.5),c(14,45))
 # Output detours around the group heading, leaving the title unobstructed.
 route(c(161,168,168,139.5,139.5),c(81.5,81.5,110,110,115))
 module(118,115,expression(paste('Decode ',f^{-1})),'#F0EEE5');route(c(161,166),c(119.5,119.5));tt(167,119.5,'Answer',size=8.2,adj=0)
 tt(10,119,'Same model; separate calls',size=9,font=2,adj=0);tt(10,112,'Source output cap: 1,024 tokens',size=8.2,adj=0)
 tt(90,3,expression(paste('Controls:  ',C+Q,' (direct)        ',C+T+Q,' (append)')),size=8.5)
}
network<-function(){n<-file.path(root,'network');v<-read.csv(file.path(n,'source_00_graph_geometry.csv'));e<-read.csv(file.path(n,'source_01_complete_graph_edges.csv'));stopifnot(nrow(v)==28,nrow(e)==112);canvas(137);hd(5,131,'a','Reversible labels preserve the complete topology')
 tt(42,121,'Original identifiers',size=9.5,font=2);tt(138,121,'Short identifiers',size=9.5,font=2)
 for(k in 1:2){cx<-c(42,138)[k];xy<-cbind(cx+v$display_x*32,82+v$display_y*25);rownames(xy)<-v$compact
  for(i in 1:nrow(e)){a<-xy[e$compact_source[i],];b<-xy[e$compact_target[i],];if(all(a==b)){th<-seq(.25,2*pi-.25,length.out=70);p<-cbind(a[1]+2+2.5*cos(th),a[2]+2.5*sin(th));lines(p,col='#B3BFC9',lwd=.55);j<-nrow(p);arrows(p[j-1,1],p[j-1,2],p[j,1],p[j,2],length=.045,col='#B3BFC9',lwd=.55)}else{u<-(b-a)/sqrt(sum((b-a)^2));arrows(a[1]+2.1*u[1],a[2]+2.1*u[2],b[1]-2.4*u[1],b[2]-2.4*u[2],length=.045,angle=20,col='#BCC7CF',lwd=.55)}}
  for(i in 1:nrow(v)){th<-seq(0,2*pi,length.out=60);polygon(xy[i,1]+2.25*cos(th),xy[i,2]+2.25*sin(th),col=if(k==1)navy else teal,border='white',lwd=.6);if(k==2)tt(xy[i,1],xy[i,2],v$compact[i],size=7.8,colour='white')}
 }
 route(c(79,100),c(82,82),navy);tt(89.5,89,expression(f),size=10)
 tt(42,48,'28 nodes / 112 directed edge entries',size=8.1);tt(138,48,'Shared positions / stored inverse mapping',size=8.1)
 segments(5,42,175,42,col=light,lwd=.6);hd(5,35,'b','The source trace is passed as text to a fresh reader')
 tt(5,25,expression(C+Q),size=9,adj=0);route(c(23,40),c(25,25));tt(57,25,'Source call',size=9);route(c(75,94),c(25,25));tt(120,25,'Native thinking T',size=9);tt(57,18,'1,024-token cap',size=8);tt(120,18,'Errors and truncation retained',size=8)
 tt(5,8,expression(paste('Reader: ',C+Q,'  |  ',T+C+Q,'  |  ',C+T+Q)),size=8.7,adj=0);route(c(124,144),c(8,8));tt(160,8,'Decode',size=8.7)
}
confirmation<-function(){src<-file.path(root,'confirmation');g<-read.csv(file.path(src,'graph_means.csv'));a<-read.csv(file.path(src,'plot_arms.csv'));ct<-read.csv(file.path(src,'plot_contrasts.csv'));arms<-c('original','original_long','compact','prefix','append');co<-c(grey,'#806C9E',navy,teal,orange);key<-g[g$arm=='compact',];key<-key[order(key$problem_type,key$graph_index),];m<-sapply(arms,function(k)g$f1[match(paste(key$graph_index,k),paste(g$graph_index,g$arm))]);delta<-m[,4]-m[,3];primary<-ct[ct$contrast=='prefix-compact',];stopifnot(nrow(m)==24,abs(mean(delta)-primary$difference)<1e-12);canvas(170)
 hd(5,163,'a','Performance distributions');tt(5,156,'24 graph means per arm',size=8,adj=0);fx<-function(z)30+z*50
 for(v in c(0,.5,1)){segments(fx(v),94,fx(v),150,col=light,lwd=.5);tt(fx(v),91,format(v),size=8)}
 for(i in 1:5){z<-g$f1[g$arm==arms[i]];y<-149-(i-1)*11;d<-density(c(-z,z,2-z),from=0,to=1,bw=.08,n=256);hh<-d$y/max(d$y)*4;polygon(c(fx(d$x),rev(fx(d$x))),c(y+hh,rep(y,length(hh))),col=adjustcolor(co[i],.3),border=NA);points(fx(z),y-1.5+rep(c(-.5,0,.5),8),pch=21,bg=co[i],col='white',cex=.55,lwd=.3);q<-quantile(z,c(.25,.5,.75));segments(fx(q[1]),y-4,fx(q[3]),y-4,col=co[i],lwd=1.8);points(fx(q[2]),y-4,pch=21,bg='white',col=co[i],cex=.55);tt(27,y-1,c('Original','Longer','Compact','Before','After')[i],size=8,adj=1)}
 tt(55,85,'Graph-mean F1',size=8.3)
 hd(97,163,'b','Paired gains and losses');tt(97,156,sprintf('+%.3f  [%.3f, %.3f]',primary$difference,primary$lower,primary$upper),size=8,adj=0)
 gx<-function(v)109+(v-1)/23*63;gy<-function(v)103+v*63
 for(v in c(-.2,0,.2,.4,.6)){segments(108,gy(v),174,gy(v),col=if(v==0)grey else light,lwd=.5);tt(105,gy(v),format(v),size=8,adj=1)}
 for(i in 1:24)rect(gx(i)-.85,gy(min(0,delta[i])),gx(i)+.85,gy(max(0,delta[i])),col=if(delta[i]>=0)teal else orange,border=NA)
 segments(gx(12.5),90,gx(12.5),148,lty=3,col=grey,lwd=.6);segments(108,gy(primary$difference),174,gy(primary$difference),lty=2,col=navy,lwd=.8)
 tt(gx(6.5),85,'BFS',size=8);tt(gx(18.5),85,'Parents',size=8);tt(96,118,'Change in F1',size=8,srt=90)
 hd(5,75,'c','Every graph and condition');pal<-colorRampPalette(c('#F5F7FA','#CEDCE7','#67A7AD','#006D77'))(101)
 for(i in 1:24)for(j in 1:5)rect(24+(j-1)*11,24+(24-i)*1.7,24+j*11-.3,24+(25-i)*1.7-.12,col=pal[1+round(m[i,j]*100)],border=NA)
 tt(20,54,'BFS',size=8,adj=1);tt(20,34,'Parents',size=8,adj=1);segments(24,44.4,79,44.4,col='white',lwd=1)
 for(j in 1:5)tt(24+(j-.5)*11,20,c('Orig.','Longer','Compact','Before','After')[j],size=7.8)
 for(i in 1:100)rect(24+(i-1)*.55,11,24+i*.55,13,col=pal[i],border=NA);tt(24,6,'F1 0',size=8);tt(51.5,6,'0.5',size=8);tt(79,6,'1',size=8)
 hd(97,75,'d','Accuracy and full cost');tt(97,68,'Before: +39.8% tokens vs compact',size=7.8,colour=teal,adj=0)
 tx<-function(z)110+(z-3000)/6000*62;ty<-function(z)19+(z-.2)/.7*42
 for(v in c(.2,.4,.6,.8)){segments(110,ty(v),172,ty(v),col=light,lwd=.5);tt(107,ty(v),format(v),size=8,adj=1)}
 for(v in c(3000,6000,9000))tt(tx(v),14,paste0(v/1000,'k'),size=8)
 z<-a[match(arms,a$arm),];segments(tx(z$mean_deployment_tokens[3]),ty(z$f1[3]),tx(z$mean_deployment_tokens[4]),ty(z$f1[4]),col=teal,lwd=1.2);points(tx(z$mean_deployment_tokens),ty(z$f1),pch=21,bg=co,col='white',cex=1)
 for(i in 1:5)tt(tx(z$mean_deployment_tokens[i])+2,ty(z$f1[i])+c(-3,3,-3,3,-3)[i],c('Original','Longer','Compact','Before','After')[i],size=7.8,colour=co[i],adj=0)
 tt(141,6,'Model tokens / answer',size=8);tt(96,40,'Mean F1',size=8,srt=90)
}
jobs<-list(pipeline=pipeline,network=network,confirmation=confirmation);hs<-c(140,137,170)
for(i in 1:3){n<-names(jobs)[i];out<-file.path(root,n);dir.create(out,recursive=TRUE,showWarnings=FALSE)
 cairo_pdf(file.path(out,'publication.pdf'),width=180/25.4,height=hs[i]/25.4,family='Arial',pointsize=10);jobs[[i]]();dev.off()
 png(file.path(out,'preview.png'),width=round(180/25.4*190),height=round(hs[i]/25.4*190),res=190,type='cairo',pointsize=10);jobs[[i]]();dev.off()
 for(f in c('publication.pdf','preview.png')){con<-file(file.path(out,f),'rb');bytes<-raw();repeat{chunk<-readBin(con,'raw',n=1048576);if(!length(chunk))break;bytes<-c(bytes,chunk)};close(con);cat('FIG_EXPORT|',n,'/',f,'|',paste(sprintf('%02x',as.integer(bytes)),collapse=''),'\n',sep='')}
}
