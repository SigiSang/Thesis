#!/usr/bin/perl
use 5.010;
use strict;
use warnings;

## Iteration parameters ##
my @params;
my @ds = ("bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet","all");

for my $i (0..@ds-1){
	$params[$i] = {
		 name => $ds[$i]
		,row_ref_init => 2+$i*7
		,line_row_skip => 2 # 2 means skip to 'sum' row, 4 means skip to 'avg' row (only for ds "all")
	};
}
$params[-1]->{line_row_skip} = 4; # change for ds "all" only


for my $par (@params){
	my ($name,$row_ref_init,$line_row_skip) = (
			 $par->{name}
			,$par->{row_ref_init}
			,$par->{line_row_skip}
		);
	##### File IO #####
	my $dir_io = "/media/tim/Game drive/Data_thesis/output/datasheets/formulas";
	open(my $csv_out,">","$dir_io/form_$name.csv") or die "Failed to open outputfile: $!";
	my $delim = ";";
	my $indirect_prefix = "=INDIRECT(\"\'stats copy\'.";

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
				push @line,"$indirect_prefix$col_let\"&$md_jmp+$ref_row_row+$line_row_skip)";
			}
			say $csv_out join $delim,@line;
		}
		print $csv_out "\n";
	}

	close($csv_out);
}