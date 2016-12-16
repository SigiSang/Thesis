#!/bin/bash
prefix=$(date +%Y-%m-%d)
fn_in_1="GanttChart.pdf"
fn_in_2="GanttChart2.pdf"
fn_out="$(date +%Y-%m-%d)_Tim_Ranson_Planning_Thesis.pdf"
page_count=$(pdfinfo "$fn_in_1" | grep 'Pages: ' | awk '{print $2}')
page_count=$(($page_count-1))
pdftk A="$fn_in_1" B="$fn_in_2" cat A1-$(($page_count-1)) B$page_count  A$page_count output "$fn_out"