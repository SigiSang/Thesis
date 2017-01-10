#!/usr/bin/perl
use 5.010;
use strict;
use warnings;

##### File IO #####
my $dir_io = "/media/tim/Game drive/Data_thesis/output/datasheets/formulas";
my $delim = ";";

for my $p ("t_sv"){
	my $prefix = "=$p.";
	open(my $csv_out,">","$dir_io/reorg_$p.csv") or die "Failed to open outputfile: $!";

	my $hdr_line = join $delim,("paramVal","noise","dsName","TP","TN","FP","FN","RE","FPR");

	my $row_init = 5;
	my $row_skip_blk = 6;
	my $row_skip_ln = 24;

	my $no_blks = 4;
	my $no_lns = 10;

	for my $blk_mult (0..$no_blks-1){
		say $csv_out $hdr_line;
		for my $ln_mult (0..$no_lns-1){
			my $row = $row_init + $blk_mult*$row_skip_blk + $ln_mult*$row_skip_ln;
			my @ln;
			for my $col ('A'..'I'){
				push @ln,"$prefix$col$row";
			}
			say $csv_out (join $delim,@ln);
		}
		print $csv_out "\n";
	}

	close($csv_out);
}