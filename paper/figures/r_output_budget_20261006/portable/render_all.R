args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else getwd(),winslash='/')
source(file.path(root,'output_budget_20261006.R'))
