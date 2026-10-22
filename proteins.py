"""The five proteins of the model-comparing dataset, as used for density guidance.

Each entry gives the BioEmu sequence (same as data/sequences_v1.txt, the sequence the earlier
unguided ensembles were generated from), the chain of the protein in the deposited PDB files,
the guidance targets (one per map), and extra deposited models used only in evaluation.

Choice of targets (from results/cross_comparison of the model-comparing project):
  GltPh    3 distinct states (outward, intermediate, inward)                      -> 3 targets
  SLC37A4  6 maps in 2 structural groups {21nq, 9kuy} and {9ugu, 9ugx, 9ws8, 9x95};
           cross CCmask between groups is about 0.05, so this is the largest transition.
           One target per group (the best resolved map), the other members are evaluation refs.
  SPNS2    2 adjacent outward-facing sub-states (smallest resolved transition)    -> 2 targets
  GPR4     same state twice (different pH)            -> 2 targets, negative control
  AUX1     same apo state, two determinations         -> 2 targets, negative control
"""

from __future__ import annotations

PROTEINS = {
    "GltPh": {
        "chain": "A",
        "sequence": (
            "MGLYRKYIEYPVLQKILIGLILGAIVGLILGHYGYAHAVHTYVKPFGDLFVRLLKMLVMPIVFASLVVGAASISPARLGRVGVKIVVYYLLTSAF"
            "AVTLGIIMARLFNPGAGIHLAVGGQQFQPHQAPPLVHILLDIVPTNPFGALANGQVLPTIFFAIILGIAITYLMNSENEKVRKSAETLLDAINGLAE"
            "AMYKIVNGVMQYAPIGVFALIAYVMAEQGVHVVGELAKVTAAVYVGLTLQILLVYFVLLKIYGIDPISFIKHAKDAMLTAFVTRSSSGTLPVTMRVA"
            "KEMGISEGIYSFTLPLGATINMDGTALYQGVCTFFIANALGSHLTVGQQLTIVLTAVLASIGTAGVPGAGAIMLAMVLHSVGLPLTDPNVAAAYAMI"
            "LGIDAILDMGRTMVNVTGDLTGTAIVAKTEGTLVPR"
        ),
        # label: (EMDB id, PDB id, resolution in A)
        "states": {
            "OFS_9bh2": ("44530", "9bh2", 2.70),
            "iOFS_9bh1": ("44529", "9bh1", 3.10),
            "IFS_6x12": ("21986", "6x12", 3.50),
        },
        "extra_refs": {},
        "role": "three distinct states",
    },
    "SLC37A4": {
        "chain": "A",
        "sequence": (
            "MAAQGYGYYRTVIFSAMFGGYSLYYFNRKTFSFVMPSLVEEIPLDKDDLGFITSSQSAAYAISKFVSGVLSDQMSARWLFSSGLLLVGLVNIFFAW"
            "SSTVPVFAALWFLNGLAQGLGWPPCGKVLRKWFEPSQFGTWWAILSTSMNLAGGLGPILATILAQSYSWRSTLALSGALCVVVSFLCLLLIHNEPA"
            "DVGLRNLDPMPSEGKKGSLKEESTLQELLLSPYLWVLSTGYLVVFGVKTCCTDWGQFFLIQEKGQSALVGSSYMSALEVGGLVGSIAAGYLSDRAM"
            "AKAGLSNYGNPRHGLLLFMMAGMTVSMYLFRVTVTSDSPKLWILVLGAVFGFSSYGPIALFGVIANESAPPNLCGTSHAIVGLMANVGGFLAGLPF"
            "STIAKHYSWSTAFWVAEVICAASTAAFFLLRNIRTKMGRVSKKAE"
        ),
        "states": {
            "grpA_21nq": ("67847", "21nq", 2.80),
            "grpB_9ugu": ("64144", "9ugu", 3.07),
        },
        # same structural group as the target named in the value; evaluation only
        "extra_refs": {"grpA_9kuy": "9kuy", "grpB_9ugx": "9ugx"},
        "role": "large transition (two structural groups)",
    },
    "SPNS2": {
        "chain": "A",
        "sequence": (
            "EKRKAAAAILSLGNVLNYLDRYTVAGVLLDIQQHFGVKDRGAGLLQSVFICSFMVAAPIFGYLGDRFNRKVILSCGIFFWSAVTFSSSFIPQQYFW"
            "LLVLSRGLVGIGEASYSTIAPTIIGDLFTKNTRTLMLSVFYFAIPLGSGLGYITGSSVKQAAGDWHWALRVSPVLGMITGTLILILVPATKRGHAD"
            "QLGDQLKARTSWLRDMKALIRNRSYVFSSLATSAVSFATGALGMWIPLYLHRAQVVQKTAETCNSPPCGAKDSLIFGAITCFTGFLGVVTGAGATR"
            "WCRLKTQRADPLVCAVGMLGSAIFICLIFVAAKSSIVGAYICIFVGETLLFSNWAITADILMYVVIPTRRATAVALQSFTSHLLGDAGSPYLIGFI"
            "SDLIRQSTKDSPLWEFLSLGYALMLCPFVVVLGGMFFLATALFFVSDRARAEQQVNQLAMPPASVKV"
        ),
        "states": {
            "s1_8ex5": ("28651", "8ex5", 3.47),
            "s2_8ex7": ("28653", "8ex7", 3.53),
        },
        "extra_refs": {},
        "role": "small transition (adjacent outward-facing sub-states)",
    },
    "GPR4": {
        "chain": "R",
        "sequence": (
            "MDNSTGTGEGCHVDSRVDHLFPPSLYIFVIGVGLPTNCLALWAAYRQVRQHNELGVYLMNLSIADLLYICTLPLWVDYFLHHDNWIHGPGSCKLFG"
            "FIFYSNIYISIAFLCCISVDRYLAVAHPLRFARLRRVKTAVAVSSVVWATELGANSAPLFHDELFRDRYNHTFCFEKFPMERWVAWMNLYRVFVGF"
            "LFPWALMLLCYRGILRAVQSSVSTERQEKVKIKRLALSLIAIVLVCFAPYHALLLSRSAVYLGRPWDCGFEERVFSAYHSSLAFTSLNCVADPILYC"
            "LVNEGARSDVAKALHNLLRFLASNKPQEMANASLTLETPLTSKRSTTGKSSGAVWAVPPTAQGDQVPLKVLLPPAQ"
        ),
        "states": {
            "pHa_8zfd": ("60057", "8zfd", 2.56),
            "pHb_9jvh": ("61839", "9jvh", 2.76),
        },
        "extra_refs": {},
        "role": "same-state control (different pH)",
    },
    "AuxinTransporter": {
        "chain": "A",
        "sequence": (
            "MSEGVEAIVANDNGTDQVNGNRTGKDNEEHDGSTGSNLSNFLWHGGSVWDAWFSCASNQVAQVLLTLPYSFSQLGMLSGIVLQIFYGLLGSWTAYL"
            "ISVLYVEYRARKEKEGKSFKNHVIQWFEVLDGLLGSYWKALGLAFNCTFLLFGSVIQLIACASNIYYINDHLDKRTWTYIFGACCATTVFIPSFHN"
            "YRIWSFLGLGMTTYTAWYLAIASIIHGQAEGVKHSGPTKLVLYFTGATNILYTFGGHAVTVEIMHAMWKPQKFKYIYLMATLYVFTLTIPSAAAVY"
            "WAFGDALLDHSNAFSLMPKNAWRDAAVILMLIHQFITFGFACTPLYFVWEKVIGMHDTKSICLRALARLPVVIPIWFLAIIFPFFGPINSAVGALL"
            "VSFTVYIIPSLAHMLTYRSASARQNAAEKPPFFMPSWTAMYVLNAFVVVWVLIVGFGFGGWASVTNFVRQVDTFGLFAKCYQCKPAAAAAHAPVSA"
            "LHHRL"
        ),
        "states": {
            "d1_9ute": ("64490", "9ute", 3.26),
            "d2_9jdr": ("61397", "9jdr", 3.50),
        },
        "extra_refs": {},
        "role": "same-state control (two independent determinations)",
    },
}

GLTPH_SEQ = PROTEINS["GltPh"]["sequence"]


def protein_of_target(path: str) -> str:
    """targets/<Protein>_<label>.pt -> <Protein>."""
    import os

    base = os.path.basename(path)
    for name in sorted(PROTEINS, key=len, reverse=True):
        if base.startswith(name + "_"):
            return name
    raise ValueError(f"cannot tell the protein from {path}")
