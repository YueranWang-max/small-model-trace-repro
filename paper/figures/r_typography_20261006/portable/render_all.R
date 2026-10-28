args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else getwd(),winslash="/")
out <- if(length(args)>1) args[2] else file.path(root,"redrawn")
script <- function(name) {code <- readLines(file.path(root,name));code <- code[-grep("^args <- commandArgs",code)];eval(parse(text=code),envir=new.env(parent=globalenv()))}
args <- c(root,out)
script("typography_20261006.R")
args <- c(root,out)
script("confirmatory_typography_20261006.R")
