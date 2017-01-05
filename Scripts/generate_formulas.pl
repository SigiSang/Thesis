#!/usr/bin/perl
use 5.010;
use strict;
use warnings;

##### File IO #####
my $dir_io = "/media/tim/Game drive/Data_thesis/output/datasheets";
open(my $csv_out,">","$dir_io/formulas.csv") or die "Failed to open outputfile: $!";
my $delim = ";";
my $indirect_prefix = "=INDIRECT(\"\'stats copy\'.";

my $row_ref_init = 44;

my ($n_pospro,$n_noise) = (2,4);
for my $iter (0..$n_pospro*$n_noise-1){
	my $ref_row_row = "\$C".(2+$iter*9);
	my $row_ref_val = "=INDIRECT(ADDRESS(ROW()-9,COLUMN()))+50";
	$row_ref_val = $row_ref_init if ($iter==0);

	say $csv_out join $delim,("pospro","noise","row ref");
	say $csv_out join $delim,(
		 $indirect_prefix."B\"&$ref_row_row)"
		,$indirect_prefix."C\"&$ref_row_row)"
		,$row_ref_val
	);
	my @hdr_line = ("md/hdr");
	for my $col_let ('B'..'L'){
		push @hdr_line,"$indirect_prefix$col_let\"&$ref_row_row+1)";
	}
	say $csv_out join $delim,@hdr_line;
	for my $row_mult (0..4){
		my $md_jmp = $row_mult*400;
		my @line = ($indirect_prefix."A\"&$md_jmp+$ref_row_row)");
		for my $col_let ('B'..'L'){
			push @line,"$indirect_prefix$col_let\"&$md_jmp+$ref_row_row+4)";
		}
		say $csv_out join $delim,@line;
	}
	print $csv_out "\n";
}

close($csv_out);