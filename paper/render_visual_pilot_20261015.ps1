$ErrorActionPreference='Stop'
$figureRoot='D:\project\tangjie\trace_as_state_repro\paper\figures\visual_pilot_20261015'
$env:TMPDIR='D:\project\tangjie\trace_as_state_repro\paper\.r-temp-distinctive-20261015'
$env:TEMP=$env:TMPDIR
$env:TMP=$env:TMPDIR
$result=& 'D:\project\tangjie\.local-r-4.6.1\bin\Rscript.exe' --vanilla 'D:\project\tangjie\trace_as_state_repro\paper\export_visual_pilot_20261015.R'
if($LASTEXITCODE -ne 0){throw 'R rendering failed'}
$count=0
foreach($line in $result){
 if($line.StartsWith('FIG_EXPORT|')){
  $parts=$line.Split('|')
  if($parts[1] -notmatch '^(method|models)/(design.pdf|design.png|preview.png)$'){throw 'Unexpected export'}
  [IO.File]::WriteAllBytes((Join-Path $figureRoot $parts[1]),[Convert]::FromHexString($parts[2]));$count++
 }else{Write-Output $line}
}
Write-Output "Transferred $count figure files"
