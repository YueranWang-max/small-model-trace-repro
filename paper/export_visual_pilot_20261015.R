source('D:/project/tangjie/trace_as_state_repro/paper/draw_visual_pilot_20261015.R')
for(n in names(jobs))for(f in c('design.pdf','design.png','preview.png')){
 con<-file(file.path(root,n,f),'rb');bytes<-raw()
 repeat{chunk<-readBin(con,'raw',n=1048576);if(!length(chunk))break;bytes<-c(bytes,chunk)}
 close(con);cat('FIG_EXPORT|',n,'/',f,'|',paste(sprintf('%02x',as.integer(bytes)),collapse=''),'\n',sep='')
}
