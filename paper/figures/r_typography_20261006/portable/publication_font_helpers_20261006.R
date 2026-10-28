# R-only publication exports at the manuscript's final scale.
stopifnot(requireNamespace('svglite',quietly=TRUE),capabilities('cairo'))
.final_font_base <- 8
.bound_cex <- function(value,relative=TRUE) {
 base <- .final_font_base * if(relative) par('cex') else 1
 pmin(6.98/base,pmax(5.25/base,value))
}
text <- function(x,...,cex=1) graphics::text(x,...,cex=.bound_cex(cex))
axis <- function(side,...,cex.axis=par('cex.axis')) graphics::axis(side,...,cex.axis=.bound_cex(cex.axis))
legend <- function(x,...,cex=1) graphics::legend(x,...,cex=.bound_cex(cex))
mtext <- function(text,...,cex=NA) {
 if(is.character(text)) text <- vapply(text,function(t) {
  if(grepl('^[A-F]  *',t)) paste0(tolower(substr(t,1,1)),substr(t,2,nchar(t))) else t
 },character(1))
 if(length(cex)==1 && is.na(cex)) cex <- 6.98/.final_font_base
 graphics::mtext(text,...,cex=.bound_cex(cex,FALSE))
}
publication_export <- function(name,draw,out,w,h,fraction=1) {
 ps <- ceiling(8*w/(6.5*fraction))
 .final_font_base <<- ps*(6.5*fraction)/w
 svglite::svglite(file.path(out,paste0(name,'.svg')),width=w,height=h,pointsize=ps,bg='white');draw();dev.off()
 grDevices::cairo_pdf(file.path(out,paste0(name,'.pdf')),width=w,height=h,family='Arial',pointsize=ps,bg='white');draw();dev.off()
 png(file.path(out,paste0(name,'.png')),width=w,height=h,units='in',res=600,type='cairo',pointsize=ps,bg='white');draw();dev.off()
 cat(name,': final fraction=',fraction,', native device pointsize=',ps,'; editable SVG, embedded PDF, R PNG\n')
}
