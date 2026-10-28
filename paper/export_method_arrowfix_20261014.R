# Preserve entire export streams, independently of cached file-size metadata.
source('D:/project/tangjie/trace_as_state_repro/paper/draw_method_pipeline_20261014.R')
for(name in c('publication_arrowfix.pdf','publication_arrowfix.png','preview_arrowfix.png')) {
 con<-file(file.path(out,name),'rb');bytes<-raw()
 repeat { chunk<-readBin(con,'raw',n=1048576);if(!length(chunk))break;bytes<-c(bytes,chunk) }
 close(con)
 cat('FIG_EXPORT|',name,'|',paste(sprintf('%02x',as.integer(bytes)),collapse=''),'\n',sep='')
}
