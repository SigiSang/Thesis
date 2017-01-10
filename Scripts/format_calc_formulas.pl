#!/usr/bin/perl
use 5.010;
use strict;
use warnings;

##### File IO #####
my $dir_io = "/media/tim/Game drive/Data_thesis/output/datasheets";
open(my $csv_out,">","$dir_io/stats.csv") or die "Failed to open outputfile: $!"; # output statistical spreadsheet

##### Spreadsheet values #####
#'file:///media/tim/Game drive/Data_thesis/output/datasheets/data_bundle.ods'#$'efic_w-pospro_noise-000_bridgeEntry'.A1
my $db = "'file://$dir_io/data_bundle.ods'";
my @heads = qw(TP TN FP FN Re Sp FPR FNR PWC F Pr);
# my @heads = qw(TP TN FP FN);
my @colLet = ('B'..'Z');
my $delim = ";";
my $first_sum_row = 4;
my $row_size_dataset = 7;
my $row_size_set = 46;

## Metric formulas ##
my ($tp_c,$tn_c,$fp_c,$fn_c,$re_c,$pr_c) = ('B'..'F','L');
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

##### Iteration parameters #####
my @md = ("efic","fbof","lobster","subsense","vibe");
my @pp = ("w","wo");
my @no = ("00".."03");
my @ds = ("bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet");

my $stats = {};

# my ($m,$p,$n,$d) = ($md[0],$pp[0],$no[0],$ds[0]);
my ($m,$p,$n,$d);
for $m (@md){
for $p (@pp){
for $n (@no){
my $d_stat = [];
for my $i (0..@heads-1){
	my $col = $i+2;# +1 for 1-indexing, +1 for extra fnc column

	# my (@tot_sum,@tot_cnt);
	my @tot_sum;
	for my $j (0..@ds-1){
		my $row = $row_size_set - ($first_sum_row+$j*$row_size_dataset);
		push @tot_sum,"INDIRECT(ADDRESS(ROW()-$row,$col,1))";
		# push @tot_cnt,"INDIRECT(ADDRESS(ROW()-$row,$col,1))";
	}

	$d_stat->[$i] = { 
		 sum => join("+",@tot_sum)
		,cnt => scalar @ds
		,avg => "INDIRECT(ADDRESS(ROW()-2,$col,1)) / INDIRECT(ADDRESS(ROW()-1,$col,1))"
	};
}
for $d (@ds){
	my $sheet = "'${m}_$p-pospro_noise-${n}0_$d'";
	my $ref = "$db\#\$$sheet.";

	my $stat = [];

	for my $i (0..@heads-1){
		my $c = "$colLet[$i]";
		my $col = $i+2; # +1 for 1-indexing, +1 for extra fnc column

		if($i<4){ # TP TN FP FN
			$stat->[$i] = {
				 sum => "SUM($ref$c:$c)"
				,cnt => "COUNT($ref$c:$c)"
			};
		}else{ # Re Sp FPR FNR PWC F Pr
			$stat->[$i] = {cnt => "INDIRECT(ADDRESS(ROW(),COLUMN()-1))"};
			$stat->[$i]->{sum} = $metric_formulas[$i-4];
		}
		$stat->[$i]->{avg} = "INDIRECT(ADDRESS(ROW()-2,$col,1)) / INDIRECT(ADDRESS(ROW()-1,$col,1))";
	}
	$stats->{$m}->{$p}->{$n}->{$d} = $stat;
}
$stats->{$m}->{$p}->{$n}->{"all"} = $d_stat;

}
}
}

for $m (@md){
for $p (@pp){
for $n (@no){
for $d (@ds,"all"){
	say $csv_out join $delim,"m","p","n","d";
	say $csv_out join $delim,"$m","$p","$n","$d";
	say $csv_out join $delim,"fnc",@heads;
	my $stat = $stats->{$m}->{$p}->{$n}->{$d};
	for my $k ("sum","cnt","avg"){
		my $tmp = "$k$delim";
		for (@$stat) {
			$tmp .= "=$_->{$k}$delim";
		}
		chop($tmp);
		say $csv_out $tmp;
	}
	say $csv_out "";
}
say $csv_out "";
}
}
}