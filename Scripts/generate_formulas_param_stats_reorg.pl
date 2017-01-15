#!/usr/bin/perl
use 5.010;
use strict;
use warnings;

##### File IO #####
my $dir_io = "/media/tim/Game drive/Data_thesis/output/datasheets/formulas";
my $delim = ";";

## Metric formulas ##
my ($tp_c,$tn_c,$fp_c,$fn_c,$re_c,$pr_c) = ('D'..'H','N');
# Recall : # tp / (tp+fn)
my $form_re = "INDIRECT(\"$tp_c\"&ROW())/(INDIRECT(\"$tp_c\"&ROW())+INDIRECT(\"$fn_c\"&ROW()))";
# Specificity : tn / (tn+fp)
my $form_sp = "INDIRECT(\"$tn_c\"&ROW())/(INDIRECT(\"$tn_c\"&ROW())+INDIRECT(\"$fp_c\"&ROW()))";
# FPR : fp / (fp+tn)
my $form_fpr = "INDIRECT(\"$fp_c\"&ROW())/(INDIRECT(\"$fp_c\"&ROW())+INDIRECT(\"$tn_c\"&ROW()))";
# FNR : fn / (tp+fn)
my $form_fnr = "INDIRECT(\"$fn_c\"&ROW())/(INDIRECT(\"$tp_c\"&ROW())+INDIRECT(\"$fn_c\"&ROW()))";
# PWC : 100 * (fp+fn) / (tp+tn+fp+fn)
my $form_pwc = "100 * (INDIRECT(\"$fp_c\"&ROW())+INDIRECT(\"$fn_c\"&ROW()))/(INDIRECT(\"$tp_c\"&ROW())+INDIRECT(\"$tn_c\"&ROW())+INDIRECT(\"$fp_c\"&ROW())+INDIRECT(\"$fn_c\"&ROW()))";
# F-score : 2*(Pr*Re) / (Pr+Re)
my $form_f = "2 * (INDIRECT(\"$pr_c\"&ROW())*INDIRECT(\"$re_c\"&ROW()))/(INDIRECT(\"$pr_c\"&ROW())+INDIRECT(\"$re_c\"&ROW()))";
# Precision : tp / (tp+fp)
my $form_pr = "INDIRECT(\"$tp_c\"&ROW())/(INDIRECT(\"$tp_c\"&ROW())+INDIRECT(\"$fp_c\"&ROW()))";

my @metric_formulas = ($form_re,$form_sp,$form_fpr,$form_fnr,$form_pwc,$form_f,$form_pr);

my %param_no_lns = (
	 minVec => 5
	,r_sn => 3
	,t_sv => 10
	,t_sn => 10
	,r_mr => 5
);

for my $p (keys %param_no_lns){ 
	my $prefix = "=$p.";
	open(my $csv_out,">","$dir_io/reorg_$p.csv") or die "Failed to open outputfile: $!";

	my $hdr_line = join $delim,(qw(paramVal noise dsName TP TN FP FN Re Sp FPR FNR PWC F Pr));

	my $row_init = 5;
	my $row_skip_blk = 6;
	my $row_skip_ln = 24;

	my $no_blks = 4;
	my $no_lns = $param_no_lns{$p};

	for my $blk_mult (0..$no_blks-1){
		say $csv_out $hdr_line;
		for my $ln_mult (0..$no_lns-1){
			my $row = $row_init + $blk_mult*$row_skip_blk + $ln_mult*$row_skip_ln;
			my @ln;
			push @ln,"$prefix$_$row" for ('A'..'G');
			push @ln, "=$_" for (@metric_formulas);
			say $csv_out (join $delim,@ln);
		}
		print $csv_out "\n";
	}

	# Print blok with avgs, see graphs for example
	say $csv_out $hdr_line;

	my $row_mult_avgs = $no_lns+2; # 1 blok = no_lns rows + 1 hdr + 1 blank line
	for my $i (0..$no_lns-1){
		my $row_init = $i+2;
		my @ln_avgs;
		push @ln_avgs,"=A$row_init","avg","all";
		for my $col ('D'..'N'){
			my @form;
			push @form,"$col".($row_init+$row_mult_avgs*$_) for (0..$no_blks-1);
			push @ln_avgs,"=AVERAGE(".(join ",",@form).")";
		}
		say $csv_out (join $delim,@ln_avgs);
	}

	close($csv_out);
}