source /home/laura/anaconda3/etc/profile.d/conda.sh
conda activate bioinformatics

cd /media/laura/MyBook/ADNI-WGS
mkdir selected_variants

for i in {1..23}; do


    echo chr$i
    bcftools view --include ID=@/home/laura/Documents/CODE/APP_genetics/genetic_profiles/data/associated_variants.txt ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr$i.vcf.gz | cut -f 1-820 > selected_variants/chr$i.vcf
    bcftools view -H selected_variants/chr$i.vcf | wc -l

    bcftools norm -Ov -m-any selected_variants/chr$i.vcf > selected_variants/chr$i.norm.vcf
    bcftools view -H selected_variants/chr$i.norm.vcf | wc -l

    vcftools --vcf selected_variants/chr$i.norm.vcf --remove-indels --max-missing 0.95 --recode --stdout > selected_variants/chr$i.filt.vcf
    bcftools view -H selected_variants/chr$i.filt.vcf | wc -l

    bcftools query -f '%CHROM %POS %ID [ %GT]\n' selected_variants/chr$i.filt.vcf > selected_variants/chr$i.tsv
    echo ''


done