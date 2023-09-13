from goatools.base import download_go_basic_obo

obo_fname = download_go_basic_obo()
from goatools.base import download_ncbi_associations

fin_gene2go = download_ncbi_associations()
from goatools.obo_parser import GODag

obodag = GODag("go-basic.obo")

from goatools.anno.genetogo_reader import Gene2GoReader

# Read NCBI's gene2go. Store annotations in a list of namedtuples
objanno = Gene2GoReader(fin_gene2go, taxids=[10090])
ns2assoc = objanno.get_ns2assc()

for nspc, id2gos in ns2assoc.items():
    print("{NS} {N:,} annotated mouse genes".format(NS=nspc, N=len(id2gos)))
from goatools.cli.ncbi_gene_results_to_python import ncbi_tsv_to_py

ncbi_tsv = "gene_result.txt"
output_py = "genes_ncbi_10090_proteincoding.py"
ncbi_tsv_to_py(ncbi_tsv, output_py)
# Compute GO annotations for hotspots
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from genes_ncbi_10090_proteincoding import GENEID2NT as GeneID2nt_mus
from goatools.base import download_ncbi_associations
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

fin_gene2go = download_ncbi_associations()
obodag = GODag("go-basic.obo")
objanno = Gene2GoReader(fin_gene2go, taxids=[10090])
ns2assoc = objanno.get_ns2assc()

goeaobj = GOEnrichmentStudyNS(
    GeneID2nt_mus.keys(),  # List of mouse protein-coding genes
    ns2assoc,  # geneid/GO associations
    obodag,  # Ontologies
    propagate_counts=False,
    alpha=0.05,  # default significance cut-off
    methods=["fdr_bh"],
)  # defult multipletest correction method

symbol_to_id = {v[5]: v[2] for v in GeneID2nt_mus.values()}
id_to_symbol = {v: k for k, v in symbol_to_id.items()}


def convert_symbol_to_id(gene_symbols):
    # TODO: consider adding warning for genes that were not encoded to id
    return [symbol_to_id[v] for v in gene_symbols if v in symbol_to_id]


def convert_id_to_symbol(ids):
    return [id_to_symbol[v] for v in ids]


def go_analysis(gene_ids, save_name, id):
    goea_results_all = goeaobj.run_study(gene_ids, prt=None)
    goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.05]
    print(
        "\tSignificant results: {E} enriched, {P} purified".format(
            E=sum(1 for r in goea_results_sig if r.enrichment == "e"),
            P=sum(1 for r in goea_results_sig if r.enrichment == "p"),
        )
    )
    save_dir = f"go_results/{save_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    goeaobj.wr_xlsx(os.path.join(save_dir, f"coex_{id}.xlsx"), goea_results_sig)
    return goea_results_all, goea_results_sig
