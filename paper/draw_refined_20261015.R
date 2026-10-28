# Evidence-led compositions; every observation is read from archived CSV data.
root<-'D:/project/tangjie/trace_as_state_repro/paper/figures/refined_r_20261015'
ink<-'#22364A';teal<-'#00848D';orange<-'#C7784E';navy<-'#365F85';grey<-'#7E8996';light<-'#DFE6EC'
getd<-function(n,k)read.csv(list.files(file.path(root,n),pattern=paste0('^source_',sprintf('%02d',k),'_'),full.names=TRUE)[1])
readmore<-function(n,f)read.csv(file.path(root,n,f))
canvas<-function(h){par(mar=rep(0,4),family='Arial',fg=ink,cex=1);plot.new();plot.window(c(0,180),c(0,h),xaxs='i',yaxs='i',asp=1)}
tt<-function(x,y,s,size=8.4,colour=ink,...)text(x,y,s,cex=size/10,col=colour,...)
hd<-function(x,y,l,s)tt(x,y,paste(l,s),size=10.2,font=2,adj=0)
num<-function(x)format(round(x),big.mark=',',trim=TRUE)
arc<-function(x,y,r0,r1,a,b,col){t<-seq(a,b,length.out=150);polygon(c(x+r1*cos(t),rev(x+r0*cos(t))),c(y+r1*sin(t),rev(y+r0*sin(t))),col=col,border='white',lwd=.6)}
ring<-function(x,y,v,cols,r=15,inner=10){th<-pi/2-c(0,cumsum(v/sum(v))*2*pi);for(i in seq_along(v))arc(x,y,inner,r,th[i],th[i+1],cols[i])}
interval<-function(xf,lo,hi,y,col=teal,est=NULL){segments(xf(lo),y,xf(hi),y,col=col,lwd=1.5);points(xf(c(lo,hi)),c(y,y),pch=21,bg='white',col=col,cex=.65);if(!is.null(est))points(xf(est),y,pch=23,bg=col,col='white',cex=.9)}
axisx<-function(x0,x1,y,lim,ticks,lab){f<-function(z)x0+(z-lim[1])/diff(lim)*(x1-x0);segments(x0,y,x1,y,col=grey,lwd=.6);for(v in ticks){segments(f(v),y,f(v),y-1.1,col=grey);tt(f(v),y-4,if(max(lim)>1000)paste0(v/1000,'k') else format(v),size=8)};tt((x0+x1)/2,y-10,lab,size=8.4);f}
cloud<-function(v,x0,x1,y,lim,col,bw,bounded=FALSE,width=7,pch=21){
 f<-function(z)x0+(z-lim[1])/diff(lim)*(x1-x0)
 if(diff(range(v))>1e-12){aug<-if(bounded)c(-v,v,2-v) else c(-v,v);d<-density(aug,from=lim[1],to=lim[2],bw=bw,n=256);h<-d$y/max(d$y)*width;polygon(c(f(d$x),rev(f(d$x))),c(y+.8+h,rep(y+.8,length(h))),col=adjustcolor(col,.27),border=col,lwd=.6)}
 yy<-if(diff(range(v))<1e-12)y+seq(-2.5,2.5,length.out=length(v)) else y-1.4+rep(c(-.7,0,.7),length.out=length(v))
 points(f(v),yy,pch=pch,bg=col,col='white',lwd=.4,cex=.7)
}
# Curves only connect the same observed graph between discrete conditions.
flow<-function(x,y,col,lwd=1){for(i in seq_len(length(x)-1)){t<-seq(0,1,length.out=80);s<-3*t^2-2*t^3;lines(x[i]+diff(x[i:(i+1)])*t,y[i]+diff(y[i:(i+1)])*s,col=col,lwd=lwd)}}

discovery<-function(){n<-'user_benefit';a<-getd(n,0);g<-readmore(n,'graph_means.csv');canvas(150)
 for(k in 1:2){ox<-if(k==1)5 else 96;st<-c('capacity','short')[k];hd(ox,143,letters[k],c('Capacity block','Short block')[k]);tt(ox,136,c('Practical capacity criterion failed','Exploratory trace comparison')[k],size=8.1,adj=0)
  for(i in 1:3){ar<-c('original','compact','prefix')[i];z<-g[g$stratum==st & g$arm==ar,];s<-a[a$stratum==st&a$arm==ar,];stopifnot(nrow(z)==12,abs(mean(z$f1)-s$f1)<1e-10);y<-c(119,100,81)[i];co<-c(grey,navy,teal)[i]
   tt(ox,y+9,c('Original','Compact','Before')[i],size=8.3,adj=0,colour=co);cloud(z$f1,ox+15,ox+73,y,c(0,1),co,.08,TRUE,width=7,pch=ifelse(z$problem_type=='bfs',22,21));tt(ox+15,y-6,sprintf('F1 %.3f  /  EM %.3f',s$f1,s$em),size=8,adj=0)
  };axisx(ox+15,ox+73,70,c(0,1),c(0,.5,1),'Graph-mean F1')
  # Three arm means; all costs use the same scale in both blocks.
  xf<-function(z)ox+13+z/18000*58;yf<-function(z)18+z*32;z<-a[a$stratum==st,];z<-z[match(c('original','compact','prefix'),z$arm),]
  tt(ox,55,'Mean quality and full cost',size=8.8,font=2,adj=0)
  segments(ox+13,18,ox+71,18,col=grey,lwd=.6);segments(ox+13,18,ox+13,50,col=grey,lwd=.6)
  for(v in c(0,.5,1))tt(ox+10,yf(v),format(v),size=8,adj=1)
  for(v in c(0,9000,18000))tt(xf(v),14,paste0(v/1000,'k'),size=8)
  for(i in 1:2)arrows(xf(z$mean_deployment_tokens[i]),yf(z$f1[i]),xf(z$mean_deployment_tokens[i+1]),yf(z$f1[i+1]),length=.065,col='#A3B1BD',lwd=.8)
  points(xf(z$mean_deployment_tokens),yf(z$f1),pch=c(21,22,23),bg=c(grey,navy,teal),col='white',cex=1.1)
  tt(ox+42,6,'Full model tokens / answer',size=8)
 }
}

replication<-function(){n<-'official_graph_confirmatory_20261006';g<-getd(n,0);a<-getd(n,1);ct<-getd(n,2);canvas(150);tt(6,106,'F1',srt=90,size=8.5)
 hd(5,143,'a','Graph-specific responses to five traces');tt(5,136,'Same graph across Direct, Before and After; five reader seeds averaged',size=8,adj=0)
 for(k in 1:2){ox<-if(k==1)14 else 103;ids<-sort(unique(g$row_index[g$problem_type==c('bfs','parents')[k]]));xs<-ox+c(8,29,50);yf<-function(v)83+v*42
  tt(ox+29,130,c('BFS queries','Parent queries')[k],size=9,font=2)
  for(v in c(0,.5,1)){segments(ox+8,yf(v),ox+50,yf(v),col=light,lwd=.6);tt(ox+4,yf(v),format(v),size=8,adj=1)}
  cc<-hcl.colors(length(ids),'Dark 3')
  for(j in seq_along(ids)){z<-g[g$row_index==ids[j],];z<-z[match(c('direct','pre','post'),z$arm),];flow(xs,yf(z$f1),adjustcolor(cc[j],.65),1);points(xs+(j-(length(ids)+1)/2)*.65,yf(z$f1),pch=21,bg=cc[j],col='white',cex=.58)}
  for(i in 1:3)tt(xs[i],78,c('Direct','Before','After')[i],size=8)
  for(j in seq_along(ids)){x<-ox+((j-1)%%3)*23;y<-71-floor((j-1)/3)*5;segments(x,y,x+3,y,col=cc[j],lwd=1.5);tt(x+4,y,ids[j],size=8,adj=0)}
 }
 hd(5,57,'b','Complete model cost');for(i in 1:3){ar<-c('direct','pre','post')[i];z<-g[g$arm==ar,];y<-c(45,31,17)[i];tt(5,y,c('Direct','Before','After')[i],adj=0,size=8.4);cloud(z$deployment_tokens,33,103,y,c(0,50000),c(grey,teal,orange)[i],2000,width=6);tt(109,y,paste(num(mean(z$deployment_tokens)),'tokens'),adj=0,size=8.4)}
 axisx(33,103,13,c(0,50000),c(0,25000,50000),'Full model tokens / answer')
}

budget<-function(){n<-'output_budget_20261006';d<-getd(n,0);old<-readmore(n,'old_graphs.csv');cs<-readmore(n,'audited_costs.csv');canvas(150);tt(6,106,'F1',srt=90,size=8.5);hd(5,143,'a','Direct-answer recovery with a larger output cap')
 tt(5,136,'Each curve joins one graph under the two caps; five seeds averaged',size=8,adj=0)
 for(k in 1:2){z<-d[d$problem_type==c('bfs','parents')[k],];z<-z[order(z$row_index),];x<-if(k==1)23 else 112;yf<-function(v)84+v*40
  tt(x+20,130,c('BFS queries','Parent queries')[k],font=2,size=9);for(v in c(0,.5,1)){segments(x,yf(v),x+40,yf(v),col=light);tt(x-4,yf(v),format(v),adj=1,size=8)}
  cc<-hcl.colors(nrow(z),'Dark 3');for(i in 1:nrow(z)){flow(c(x,x+40),yf(c(z$old_f1[i],z$f1[i])),adjustcolor(cc[i],.7),1.2);points(c(x,x+40)+(i-(nrow(z)+1)/2)*.65,yf(c(z$old_f1[i],z$f1[i])),pch=21,bg=cc[i],col='white',cex=.65)
   xx<-x-3+((i-1)%%3)*21;yy<-72-floor((i-1)/3)*5;segments(xx,yy,xx+3,yy,col=cc[i],lwd=1.5);tt(xx+4,yy,z$row_index[i],size=8,adj=0)
  };tt(x,79,'8,192',size=8);tt(x+40,79,'28,672',size=8)
 }
 hd(5,57,'b','Realized cost, including complete preparation')
 vals<-list(old$deployment_tokens[old$arm=='direct'],d$deployment_tokens,old$deployment_tokens[old$arm=='post'],old$deployment_tokens[old$arm=='pre'])
 for(i in 1:4){stopifnot(abs(mean(vals[[i]])-cs$tokens[i])<1e-8);y<-c(45,34,23,12)[i];tt(5,y,c('Direct 8,192','Direct 28,672','Trace after','Trace before')[i],adj=0,size=8.1);cloud(vals[[i]],43,128,y,c(0,65000),c(grey,navy,orange,teal)[i],2000,width=4.5);tt(133,y,paste(num(cs$tokens[i]),'tokens'),adj=0,size=8.1)}
 tt(43,3,'0',size=8);tt(128,3,'65k',size=8);tt(85,3,'Full model tokens / answer',size=8)
}

sourceplot<-function(){d<-getd('source_qualification_figure',0);d<-d[d$phase=='scored_source',];canvas(140);hd(5,133,'a','Generation cost varies within the same graph');tt(5,126,'Paired seeds; labels are graph identifiers. Colour indicates observed F1.',size=8,adj=0)
 xf<-function(v)23+v/12000*133;yf<-function(v)36+v/30000*78
 for(v in c(0,10000,20000,30000)){segments(23,yf(v),156,yf(v),col=light,lwd=.6);tt(19,yf(v),paste0(v/1000,'k'),adj=1,size=8)}
 axisx(23,156,36,c(0,12000),c(0,4000,8000,12000),'Model input tokens');tt(7,77,'Generated tokens',srt=90,size=8.5)
 ids<-sort(unique(d$row_index));for(id in ids){z<-d[d$row_index==id,];co<-ifelse(z$f1==1,teal,orange);segments(xf(z$prompt_tokens[1]),yf(z$output_tokens[1]),xf(z$prompt_tokens[2]),yf(z$output_tokens[2]),col='#9AAEBB',lwd=1.2);points(xf(z$prompt_tokens),yf(z$output_tokens),pch=c(21,24),bg=co,col='white',lwd=.5,cex=1.1)}
 # Labels use fixed non-overlapping placements; connectors identify observations.
 lx<-c(50,50,120,120,50,50,120,120);ly<-c(44,52,50,66,61,79,103,86)
 for(i in seq_along(ids)){z<-d[d$row_index==ids[i],];mx<-mean(xf(z$prompt_tokens));my<-mean(yf(z$output_tokens));segments(mx,my,lx[i],ly[i],col='#B6C2CB',lwd=.5);tt(lx[i],ly[i],ids[i],size=8.5,font=2)}
 hd(5,16,'b','Correctness');tt(42,16,'14/16 readings: F1 = EM = 1',size=8.3,adj=0)
 z<-d[d$row_index==504,];tt(5,7,sprintf('Graph 504: F1 %.3f / %.3f; EM 0 / 0',z$f1[1],z$f1[2]),size=8.2,adj=0)
 points(c(123,145),c(8,8),pch=c(21,24),bg=teal,col='white',cex=.9);tt(127,8,'Seed 1',size=8,adj=0);tt(149,8,'Seed 2',size=8,adj=0)
}

interrupted<-function(){n<-'interrupted_study_figure';b<-getd(n,0);inp<-getd(n,1);cov<-getd(n,2);canvas(155);hd(5,148,'a','Execution coverage');ring(32,115,cov$slots,c(teal,orange,navy,light),r=22,inner=14);tt(32,117,'480',size=17,font=2);tt(32,109,'planned slots',size=8)
 labs<-c('Qualified','Budget-declined','Unqualified','Unreached');for(i in 1:4){y<-134-(i-1)*10;segments(61,y+2,61,y-2,col=c(teal,orange,navy,light)[i],lwd=3);tt(66,y,paste(cov$slots[i],labs[i]),adj=0,size=8.4)}
 hd(5,82,'b','Measured inputs and serving limits');xf<-function(v)39+v/160000*127
 for(v in c(98240,131072))segments(xf(v),46,xf(v),74,col=if(v==98240)grey else orange,lty=3,lwd=1)
 for(i in 1:5){y<-73-(i-1)*6;tt(5,y,c('Direct','Question first','Re2','Before','After')[i],adj=0,size=8);segments(xf(0),y,xf(inp$input_tokens[i]),y,col=light,lwd=1);points(xf(inp$input_tokens[i]),y,pch=23,bg=if(i<4)teal else orange,col='white',cex=.85);tt(xf(inp$input_tokens[i])+3,y,num(inp$input_tokens[i]),size=8,adj=0)}
 tt(xf(98240),41,'98,240 input',size=8);tt(xf(131072),36,'131,072 context',size=8)
 hd(5,29,'c','F1 identification bounds');for(i in 1:8){ox<-if(i<=4)5 else 96;y<-22-((i-1)%%4)*5;tt(ox,y,gsub('_',' ',b$arm[i]),adj=0,size=8);ff<-function(v)ox+28+v*20;interval(ff,b$lower[i],b$upper[i],y,navy);tt(ox+51,y,sprintf('[%.3f, %.3f]',b$lower[i],b$upper[i]),size=8,adj=0)}
}

missing<-function(n,indexed=FALSE){d<-getd(n,0);h<-if(indexed)150 else 155;canvas(h);hd(5,h-7,'a',if(indexed)'Observed coverage and remaining uncertainty' else 'Early-state outcomes remain partially identified')
 tt(5,h-16,'Open endpoints delimit identification bounds, not confidence intervals.',size=8.1,adj=0)
 tt(5,h-25,'Reader arm',size=8,font=2,adj=0);tt(46,h-25,'Known',size=8,font=2);tt(97,h-25,if(indexed)'F1 (teal) / EM (orange)' else 'F1 bounds',size=8,font=2);tt(157,h-25,'Model tokens',size=8,font=2)
 ys<-seq(h-40,29,length.out=nrow(d));fx<-function(v)65+v*67
 for(i in 1:nrow(d)){y<-ys[i];tt(5,y,gsub('_',' ',d$arm[i]),size=8.1,adj=0);ring(46,y,c(d$known[i],d$planned[i]-d$known[i]),c(teal,light),r=5.5,inner=4.3);tt(46,y,sprintf('%d/%d',d$known[i],d$planned[i]),size=8)
  interval(fx,d$f1_lower[i],d$f1_upper[i],y+if(indexed)2 else 0,teal)
  if(indexed)interval(fx,d$em_lower[i],d$em_upper[i],y-2,orange)
  tt(98,y-6,sprintf('[%.3f, %.3f]',d$f1_lower[i],d$f1_upper[i]),size=8)
  v<-if(indexed)d$tokens_lower[i] else d$planned_prediction_tokens_lower[i];complete<-if(indexed)as.logical(d$cost_complete[i]) else FALSE
  tt(157,y,paste0(if(complete)'' else '>= ',num(v)),size=8.4)
 }
 axisx(65,132,18,c(0,1),c(0,.5,1),'Score over all planned outcomes')
 tt(5,4,if(indexed)'Token sums; >= denotes an incomplete total.' else 'Known charged tokens / 32 planned predictions; all costs incomplete.',size=8,adj=0)
}

identity<-function(){n<-'official_graph_identity_recode';a<-getd(n,0);d<-getd(n,1);canvas(150);tt(6,106,'F1',srt=90,size=8.5);hd(5,143,'a','Reversible labels change graph-level responses');tt(5,136,'Every graph is retained; two reader seeds averaged',size=8,adj=0)
 for(k in 1:2){z<-d[d$type==c('bfs','parents')[k],];z<-z[order(z$row_index),];ox<-if(k==1)18 else 109;xs<-ox+c(0,22,44);yf<-function(v)81+v*44;tt(ox+22,130,c('BFS queries','Parent queries')[k],size=9,font=2)
  for(v in c(0,.5,1)){segments(xs[1],yf(v),xs[3],yf(v),col=light);tt(ox-4,yf(v),format(v),size=8,adj=1)}
  cc<-hcl.colors(nrow(z),'Dark 3');for(i in 1:nrow(z)){yy<-as.numeric(z[i,c('original_f1','compact_f1','long_alias_f1')]);flow(xs,yf(yy),adjustcolor(cc[i],.8),1.2);points(xs+(i-(nrow(z)+1)/2)*.65,yf(yy),pch=21,bg=cc[i],col='white',cex=.7);xx<-ox-5+((i-1)%%2)*32;yt<-70-floor((i-1)/2)*6;segments(xx,yt,xx+3,yt,col=cc[i],lwd=1.5);tt(xx+4,yt,z$row_index[i],size=8,adj=0)}
  for(i in 1:3)tt(xs[i],76,c('Original','Compact','Long alias')[i],size=8)
 }
 hd(5,55,'b','Input and output accounting');for(i in 1:3){x<-c(31,90,149)[i];ring(x,32,c(a$input_tokens[i],a$output_tokens[i]),c(navy,teal),r=14,inner=10);tt(x,33,num(a$tokens[i]),size=10.5,font=2);tt(x,27,'tokens',size=8);tt(x,49,c('Original','Compact','Long alias')[i],size=8.5,font=2);tt(x,12,sprintf('F1 %.3f [%.3f, %.3f]',a$f1[i],a$f1_lo[i],a$f1_hi[i]),size=8)}
 tt(5,3,'Angle = input/output share; totals printed. Blue: input. Teal: output.',size=8,adj=0)
}

reasoning<-function(){n<-'official_graph_reasoning_mode';a<-getd(n,0);d<-getd(n,1);canvas(145);tt(6,77,'Mean F1',srt=90,size=8.5);hd(5,138,'a','Quality and cost within task-length strata');tt(5,131,'Arrow: thinking off to on; four graphs per stratum',size=8.1,adj=0)
 groups<-unique(d[c('type','length_bin')]);groups<-groups[order(groups$type,groups$length_bin),];cc<-c(teal,navy,orange,'#80669B')
 for(k in 1:4){ox<-if(k<=2)22 else 111;oy<-if(k%%2==1)87 else 35;z<-d[d$type==groups$type[k]&d$length_bin==groups$length_bin[k],];z<-z[match(c('off','on'),z$arm),];fx<-function(v)ox+v/45000*48;fy<-function(v)oy+v*28
  for(v in c(0,.5,1)){segments(ox,fy(v),ox+48,fy(v),col=light,lwd=.6);tt(ox-4,fy(v),format(v),size=8,adj=1)}
  segments(ox,oy,ox+48,oy,col=grey,lwd=.6);for(v in c(0,20000,40000))tt(fx(v),oy-4,paste0(v/1000,'k'),size=8)
  tt(ox,oy+34,paste(if(groups$type[k]=='bfs')'BFS' else 'Parents',groups$length_bin[k]),size=8.6,font=2,adj=0)
  arrows(fx(z$deployment_tokens[1]),fy(z$f1[1]),fx(z$deployment_tokens[2]),fy(z$f1[2]),length=.09,col=cc[k],lwd=1.6);points(fx(z$deployment_tokens),fy(z$f1),pch=c(21,23),bg=c('white',cc[k]),col=cc[k],cex=.9)
 }
 tt(90,20,'Stratum model tokens (sum of seed-averaged costs)',size=8.2)
 hd(5,12,'b','All 16 graphs');for(i in 1:3)tt(46+(i-1)*47,10,paste(c('On','Off','Fixed route')[i],num(a$deployment_tokens[i])),size=8.3)
 for(i in 1:3)tt(46+(i-1)*47,3,sprintf('F1 %.3f [%.3f, %.3f]',a$f1[i],a$f1_lo[i],a$f1_hi[i]),size=8)
}

jobs<-list(user_benefit=discovery,official_graph_confirmatory_20261006=replication,output_budget_20261006=budget,source_qualification_figure=sourceplot,interrupted_study_figure=interrupted,partial_state_display=function()missing('partial_state_display'),interrupted_indexed_figure=function()missing('interrupted_indexed_figure',TRUE),official_graph_identity_recode=identity,official_graph_reasoning_mode=reasoning)
heights<-c(150,150,150,140,155,155,150,150,145)
for(i in seq_along(jobs)){n<-names(jobs)[i];out<-file.path(root,n);h<-heights[i]
 cairo_pdf(file.path(out,'publication.pdf'),width=180/25.4,height=h/25.4,family='Arial',pointsize=10);jobs[[i]]();dev.off()
 png(file.path(out,'publication.png'),width=round(180/25.4*300),height=round(h/25.4*300),res=300,type='cairo',pointsize=10);jobs[[i]]();dev.off()
 png(file.path(out,'preview.png'),width=round(180/25.4*170),height=round(h/25.4*170),res=170,type='cairo',pointsize=10);jobs[[i]]();dev.off()
 cat('DRAWN ',n,'\n',sep='')
}
