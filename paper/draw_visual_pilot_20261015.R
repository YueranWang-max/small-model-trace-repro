# Two evidence-led compositions. R is the exclusive drawing/export backend.
root<-'D:/project/tangjie/trace_as_state_repro/paper/figures/visual_pilot_20261015'
ink<-'#22364A';teal<-'#00848D';orange<-'#C7784E';navy<-'#365F85';grey<-'#7E8996'
read<-function(n,f)read.csv(file.path(root,n,f))
canvas<-function(h){par(mar=rep(0,4),family='Arial',fg=ink,cex=1);plot.new();plot.window(c(0,180),c(0,h),xaxs='i',yaxs='i',asp=1)}
tt<-function(x,y,s,size=8.5,colour=ink,...)text(x,y,s,cex=size/10,col=colour,...)
heading<-function(x,y,l,s)tt(x,y,paste(l,s),size=10.5,font=2,adj=0)
arcband<-function(cx,cy,r0,r1,a,b,col){th<-seq(a,b,length.out=140);polygon(c(cx+r1*cos(th),rev(cx+r0*cos(th))),c(cy+r1*sin(th),rev(cy+r0*sin(th))),col=col,border='white',lwd=1.2)}
circle<-function(x,y,r,col){th<-seq(0,2*pi,length.out=100);polygon(x+r*cos(th),y+r*sin(th),col=col,border='white',lwd=.7)}
edge<-function(a,b,r,col){v<-b-a;u<-v/sqrt(sum(v^2));arrows(a[1]+r*u[1],a[2]+r*u[2],b[1]-(r+.5)*u[1],b[2]-(r+.5)*u[2],col=col,lwd=1.2,length=.075,angle=22)}
draw_method<-function(){
 d<-read('method','dictionary.csv');cst<-read('method','components.csv');inp<-read('method','inputs.csv')
 stopifnot(abs(sum(cst$mean_tokens)-5344.458333333333)<1e-9)
 canvas(140);heading(5,133,'a','Exact reversible substitutions');heading(95,133,'b','Preparation-inclusive cost')
 tt(5,125,'Graph 87: three-vertex, two-edge excerpt',size=8,adj=0)
 pos<-rbind(V=c(14,98),G=c(34,110),Z=c(34,86))
 edge(pos['V',],pos['G',],1.7,grey);edge(pos['V',],pos['Z',],1.7,grey)
 for(i in rownames(pos))circle(pos[i,1],pos[i,2],1.7,navy)
 tt(11,104,d$original[match('V',d$compact)],size=7.8)
 tt(34,116,d$original[match('G',d$compact)],size=7.8)
 tt(34,80,d$original[match('Z',d$compact)],size=7.8)
 pos2<-sweep(pos,2,c(46,0),'+')
 edge(pos2['V',],pos2['G',],3.5,teal);edge(pos2['V',],pos2['Z',],3.5,teal)
 for(i in rownames(pos2)){circle(pos2[i,1],pos2[i,2],3.5,teal);tt(pos2[i,1],pos2[i,2],i,size=9,colour='white',font=2)}
 arrows(42,98,51,98,col=ink,lwd=1.1,length=.09);tt(46.5,104,expression(f),size=10)
 tt(13,73,'Original labels',size=8);tt(71,73,'Compact labels',size=8)
 tt(5,64,'Stored inverse dictionary',font=2,size=8.8,adj=0)
 for(i in 1:3)tt(5,57-(i-1)*7,paste0(d$compact[i],'  ->  ',d$original[i]),size=8.5,adj=0)
 tt(5,34,'All source elements remain available.',size=8,adj=0)
 # Exact hierarchical partition: inner source/reader, outer input/output.
 v<-cst$mean_tokens;total<-sum(v);starts<-pi/2-c(0,cumsum(v/total)*2*pi)
 cx<-133;cy<-92;cc<-c('#BDD1E3',navy,teal,'#D69A73')
 arcband(cx,cy,10,20,starts[1],starts[3],navy)
 arcband(cx,cy,10,20,starts[3],starts[5],teal)
 for(i in 1:4)arcband(cx,cy,22,28,starts[i],starts[i+1],cc[i])
 tt(cx,94,'5,344',size=13,font=2);tt(cx,88,'tokens',size=8)
 tt(97,124,sprintf('Source %.1f%%',sum(v[1:2])/total*100),size=8,colour=navy,adj=0)
 tt(150,124,sprintf('Reader %.1f%%',sum(v[3:4])/total*100),size=8,colour=teal,adj=0)
 for(i in 1:4){x<-if(i%%2==1)98 else 140;y<-if(i<=2)55 else 42
  segments(x-3,y+1,x-3,y-5,col=cc[i],lwd=3)
  tt(x,y,cst$component[i],size=8.1,adj=0)
  tt(x,y-5,sprintf('%s  (%.1f%%)',format(round(v[i]),big.mark=','),v[i]/total*100),size=8.2,adj=0)
 }
 tt(95,30,'Means: 24 sources, 48 prefix readers',size=7.5,adj=0)
 segments(5,27,175,27,col='#D9E1E7',lwd=.6)
 heading(5,22,'c','Measured reader input: same graph, first seed')
 for(i in 1:3){x<-c(7,71,134)[i];tt(x,15,c('Compact direct','Trace before','Trace after')[i],size=8.3,adj=0);tt(x,7,format(inp$input_tokens[i],big.mark=','),size=15,font=2,colour=c(navy,teal,orange)[i],adj=0);tt(x+25,7,'tokens',size=8,adj=0)}
}
draw_models<-function(){
 g<-list(read('models','graphs_17.csv'),read('models','graphs_4.csv'))
 a<-list(read('models','arms_17.csv'),read('models','arms_4.csv'))
 ct<-list(read('models','contrasts_17.csv'),read('models','contrasts_4.csv'))
 arms<-c('pre','post','direct');cols<-c(teal,orange,grey);labs<-c('Before','After','Direct')
 for(k in 1:2)for(arm in arms){z<-g[[k]][g[[k]]$arm==arm,];s<-a[[k]][a[[k]]$arm==arm,];stopifnot(nrow(z)==12,abs(mean(z$f1)-s$f1)<1e-12,abs(mean(z$em)-s$em)<1e-12,abs(mean(z$deployment_tokens)-s$mean_deployment_tokens)<1e-8)}
 canvas(160);heading(5,153,'a','Answer quality');heading(105,153,'b','Complete model cost')
 tt(5,146,'12 graphs per arm, five reader seeds; dashed F1 reference = always-empty baseline (1/6)',size=8,adj=0)
 fx<-function(z)32+z*56;tx<-function(z)112+z/65000*58
 for(x in c(0,.5,1)){segments(fx(x),48,fx(x),139,col='#E7ECF0',lwd=.5);tt(fx(x),141,format(x),size=8)}
 segments(fx(1/6),48,fx(1/6),139,col='#B5C0C9',lwd=.7,lty=3)
 for(x in c(0,20000,40000,60000)){segments(tx(x),48,tx(x),139,col='#E7ECF0',lwd=.5);tt(tx(x),141,paste0(x/1000,'k'),size=8)}
 denf<-list();dent<-list()
 for(k in 1:2)for(i in 1:3){z<-g[[k]][g[[k]]$arm==arms[i],];key<-paste(k,i)
  if(diff(range(z$f1))>1e-12)denf[[key]]<-density(c(-z$f1,z$f1,2-z$f1),from=0,to=1,bw=.08,n=256)
  dent[[key]]<-density(c(-z$deployment_tokens,z$deployment_tokens),from=0,to=65000,bw=2000,n=256)
 }
 maxf<-max(sapply(denf,function(z)max(z$y)));maxt<-max(sapply(dent,function(z)max(z$y)))
 for(k in 1:2){tt(5,c(137,87)[k],c('Qwen3-1.7B','Qwen3-4B')[k],font=2,size=8.7,adj=0)
  for(i in 1:3){y<-c(130,116,102,80,66,52)[(k-1)*3+i];z<-g[[k]][g[[k]]$arm==arms[i],];z<-z[order(z$row_index),];s<-a[[k]][a[[k]]$arm==arms[i],];key<-paste(k,i)
   tt(5,y,labs[i],size=8.5,colour=cols[i],adj=0)
   if(key %in% names(denf)){d<-denf[[key]];h<-d$y/maxf*5;polygon(c(fx(d$x),rev(fx(d$x))),c(y+.5+h,rep(y+.5,length(h))),col=adjustcolor(cols[i],.35),border=cols[i],lwd=.6)}
   # Exact ceiling mass is shown as observations, never as a smoothed distribution.
   yy<-if(diff(range(z$f1))<1e-12)y+seq(-2.3,2.3,length.out=12) else y-1.5+rep(c(-.65,0,.65),4)
   points(fx(z$f1),yy,pch=ifelse(z$problem_type=='bfs',22,21),bg=cols[i],col='white',lwd=.4,cex=.68)
   tt(32,y-6.5,sprintf('F1 %.3f  |  EM %.3f',s$f1,s$em),size=7.9,adj=0)
   d<-dent[[key]];h<-d$y/maxt*5;polygon(c(tx(d$x),rev(tx(d$x))),c(y+.5+h,rep(y+.5,length(h))),col=adjustcolor(cols[i],.35),border=cols[i],lwd=.6)
   points(tx(z$deployment_tokens),y-1.5+rep(c(-.65,0,.65),4),pch=ifelse(z$problem_type=='bfs',22,21),bg=cols[i],col='white',lwd=.4,cex=.68)
   tt(112,y-6.5,paste('Mean',format(round(s$mean_deployment_tokens),big.mark=','),'tokens'),size=7.9,adj=0)
  }
 }
 segments(5,41,175,41,col='#D9E1E7',lwd=.6)
 heading(5,36,'c','Paired F1 effects');tt(83,36,'Existing descriptive 95% intervals',size=8,adj=0)
 for(k in 1:2){x0<-if(k==1)33 else 122;x1<-x0+52;bx<-function(z)x0+(z+.1)/.6*(x1-x0);labx<-if(k==1)5 else 94
  tt((x0+x1)/2,29,c('1.7B','4B')[k],size=8.5,font=2)
  segments(bx(0),7,bx(0),25,col='#ADB9C3',lty=3,lwd=.7)
  for(i in 1:2){y<-c(22,10)[i];z<-ct[[k]][i,];tt(labx,y,c('Before - after','Before - direct')[i],size=7.8,adj=0)
   segments(bx(z$lower),y,bx(z$upper),y,col=if(k==1)navy else teal,lwd=1.2)
   segments(bx(z$lower),y-1,bx(z$lower),y+1,col=grey,lwd=.7);segments(bx(z$upper),y-1,bx(z$upper),y+1,col=grey,lwd=.7)
   points(bx(z$mean_difference),y,pch=23,bg=if(k==1)navy else teal,col='white',cex=.9)
   tt((x0+x1)/2,y-4.5,sprintf('%+.3f  [%.3f, %.3f]',z$mean_difference,z$lower,z$upper),size=7.8)
  }
 }
}
jobs<-list(method=draw_method,models=draw_models)
for(n in names(jobs)){
 out<-file.path(root,n);h<-if(n=='method')140 else 160
 cairo_pdf(file.path(out,'design.pdf'),width=180/25.4,height=h/25.4,family='Arial',pointsize=10);jobs[[n]]();dev.off()
 png(file.path(out,'design.png'),width=round(180/25.4*400),height=round(h/25.4*400),res=400,type='cairo',pointsize=10);jobs[[n]]();dev.off()
 png(file.path(out,'preview.png'),width=round(180/25.4*220),height=round(h/25.4*220),res=220,type='cairo',pointsize=10);jobs[[n]]();dev.off()
 cat('DRAWN ',n,'\n',sep='')
}
