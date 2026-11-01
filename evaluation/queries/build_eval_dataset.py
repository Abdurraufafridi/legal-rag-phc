#!/usr/bin/env python
"""Build the versioned legal retrieval evaluation dataset.

The relevance labels in this file were assigned by manual review of each
judgment's headnote and text during the audit. They are PROVISIONAL: the FYP
student must verify them before treating them as legally authoritative.

Run:
    ./bin/python evaluation/queries/build_eval_dataset.py

Outputs:
    evaluation/queries/evaluation_queries.jsonl
    evaluation/queries/relevance_judgments.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET_VERSION = "v1.0.0"
CREATED = "2026-10-27"
LABEL_SOURCE = "manual review of judgment headnote/text (provisional)"
ROOT = Path(__file__).resolve().parents[2]

DOC = {
    "naseeb_khan_2012": "CrM BANo240-D2012 Naseeb Khan vs State.txt",
    "parke_davis_2010": "CrA No 120 of 2008 MS Parke Davis  Co Ltd vs Provincial Drug Inspector etc.txt",
    "ismail_khan_narcotics": "Cr A No257  2612010 Ismail Khan vs  State.txt",
    "sami_ud_din_cheque": "CrM QP No1622012 Sami-ud-Din vs Nadeem Ahmad.txt",
    "state_v_wahid": "CrANo3392013 State vs Wahid.txt",
    "jehanzada_murder": "Cr M BA No1021-P2012 Jehanzada vs State.txt",
    "manzoor_s188": "Cr M QP No1762010 Manzoor Ahmad vs state.txt",
    "shah_behram_murder": "CrM BA No942013 Shah Behram vs State.txt",
    "kashif_sentence": "Cr A No3002010 Kashif vs State.txt",
    "yasmeen_narcotics": "Cr A No162009 Mst Yasmeen vs State.txt",
    "state_v_abdul_sattar": "Cr M BCA No32-D2012 State vs Abdul Sattar.txt",
    "amanullah_narcotics": "CrM BA No 51-D2012 Amanullah vs State.txt",
    "basraj_bibi_cancel": "Cr M BCANo4242010 Basraj Bibi vs Ibraheem.txt",
    "pinda_khan_ransom": "CrM BA No1312013 Pinda Khan vs  State.txt",
    "shafqat_s540": "Cr R No322011 Shafqatullah Khan vs Jehan Zeb Khan.txt",
    "naseeb_rahman_vehicle": "CrA No 78-M of 2013 Naseeb Rahman Vs The State.txt",
    "tahir_liquor": "CrM BA No16002009 Tahir vs State.txt",
    "sher_muhammad_acquittal": "Cr A No582006 Sher Muhammad vs Muhammad Akber.txt",
    "allah_baksh_fraud": "Cr M QP No154 CrM No15502011 Allah Baksh vs State.txt",
    "yasir_rehan_juvenile": "CrMBA No4422012 Yasir Rehan vs State etc.txt",
    "gul_zareen_water": "CrA No402011 Gul Zareen vs Hamzada.txt",
    "nazir_ahmad_forgery": "Cr A No992011 Nazir Ahmad vs Shaukat Zaman.txt",
    "khalid_khan_narcotics": "CrM BA No967-P2012 Khalid Khan vs  State.txt",
    "yasir_arafat_driving": "Cr A No5862011 Yasir Arafat vs  State.txt",
    "sahib_jan_abduction": "CrM BA No44-D2012 Sahib Jan vs State.txt",
    "salih_muhammad_vehicle": "Cr A No6242011 Salih Muhammad vs Niaz Muhammad.txt",
    "sharifa_bibi_narcotics": "Cr M BA No163-P2012 Sharifa Bibi vs State.txt",
    "wahabuddin_cancel": "CrM BCANo14222009 Wahabuddin vs Fayaz Khan.txt",
    "said_muhammad_cancel": "Cr M No293-M  BCA No1-M2013 Said Muhammad vs Muahammad Siraj ud Din etc.txt",
    "mehmood_shah_murder": "CrM BA No14512010 Mehmood Shah vs Noor Zada etc.txt",
    "said_rehman_murder": "CrM BA No15912009 Said Rehman vs State.txt",
    "khyal_nawaz_324": "CrMBANo5862010 Khyal Nawaz vs State.txt",
    "ishfaq_cancel": "CrM BCA No4802011 Muhammad Ishfaq vs Sher Ahmad.txt",
    "faiz_wahab_murder": "Cr M BA No7142011 Faiz ul Wahab vs State.txt",
    "ismail_khan_abduction": "CrMBA No430-P2013 Ismail Khan vs State.txt",
    "kaleem_driving": "Cr Misc B Application No 27-D of 2013 Kaleem Ullah VS The State.txt",
    "yousaf_dispossession": "WPNo18432010 Yousaf vs Bahader Sher.txt",
    "ibrahim_narcotics": "CrM BA No1603 of 2010 Ibrahim vs State.txt",
    "aftab_mpo": "WPNo232012 Aftab Khan vs Govt of KPK.txt",
    "shoaib_rape": "CrM BA No385-M2012 Muhammad Shoaib vs State.txt",
    "abdul_ahad_gambling": "CrM QP No74-D2013 Abdul Ahad vs State.txt",
    "riaz_narcotics": "CrM BANo2602012 Muhammad Riaz vs State.txt",
    "mir_tiaz_caste": "Cr M QP No462011 Mir Tiaz Khan vs  State.txt",
    "mumtaz_s540": "CrR No382012 State through Muslim vs Mumtaz Ali.txt",
    "yasir_khan_quash": "Cr MiscQuashment No14-A 2011 Yasir Khan vs Imtiaz etc.txt",
    "asher_nab": "Accouintability Appeal No 322010 Asheer Waseem Bhati vs NAB.txt",
    "irfan_theft": "Cr M BA No472012 Irfan vs Shah Jehan.txt",
    "rafi_ullah_324": "Cr M BA No4-D2011 Rafi Ullah vs State.txt",
    "fayaz_cheque": "CrM BA No16972009 Muhammad Fayaz vs State.txt",
    "wahab_ransom": "CrM BANo15782009 Wahab vs State.txt",
    "fazal_amin_currency": "CrM BA No1006-P2012 Fazal Amin vs State.txt",
    "sherin_weapons": "CrM BA  No925-P2012 Muhammad Sherin vs  State.txt",
    "qadar_police": "Cr M QP No44-B2011 Qadar Ullah vs SHO.txt",
    "munir_murder": "CrMBA No16792009 MMunir vs MIshaq.txt",
    "hussain_murder": "CrM BA No 60-D2013 Hussain Khan vs State.txt",
    "salam_bibi_acquittal": "CrA No 62-B2004 Mst Salam Bibi vs Aman Ullah.txt",
    "nasimullah_dispossession": "Cr M QP No612011 Nasimullah vs  Ijaz Hussain.txt",
    "afsar_police": "Cr M QPNo5-B2011 Afsar Ali Vs SP Bannu etc.txt",
    "khan_zeb_theft": "CrMNo16392009 Khan Zeb vs State.txt",
    "shahzeb_324": "CrM No687-A2012 Shahzeb Khan vs State.txt",
    "akram_murder": "Cr M BA No276-M2013 Muhammad Akram etc vs State etc.txt",
    "ahmad_cancel": "Cr M BCA No18022012 Ahmad Ali Jan vs Irfan.txt",
    "salam_noor_narcotics": "CrMBA No156-P2013 Salam Noor vs State.txt",
    "hafiz_shadoo_496a": "CrM QP No312012  Hafiz Shadoo vs Abdul Ghaffar.txt",
    "luqman_murder": "Cr M BA No1240-P2012 Luqman vs Daud etc.txt",
    "abid_habeas": "CrR No34-A2013 Muhammad Abid Khan vs Mst Noreen.txt",
    "inamullah_murder": "CrM BANo17942009 Inamullah vs State.txt",
    "afzal_narcotics": "CrM BA No17462011 Muhammad Afzal vs State.txt",
    "ashraf_murder_appeal": "CrA 134 to 137 MRNo12 CrR No36 2010 Muhammad Ashraf vs  Javed etc.txt",
    "fakhre_324": "CrM BA No1079-P2012 Fakhre Alam vs State.txt",
    "imran_weapons": "BANo2212013 Imran Khan vs State.txt",
    "shaukat_336": "CrM BANo17172009 Shaukat vs State.txt",
    "ibrar_336": "CrM BANo614-A2013 Muahmmad Ibrar vs State.txt",
    "noshad_narcotics": "CrM BA846-P2012 Noshad vs State.txt",
}

QUERIES = [
    # Narcotics
    ("Q001", "bail for a driver caught with charas hidden in the secret cavities of the vehicle", "bail_narcotics", [
        ("khalid_khan_narcotics", 3), ("salam_noor_narcotics", 3),
        ("amanullah_narcotics", 2), ("afzal_narcotics", 2)]),
    ("Q002", "is a woman passenger entitled to bail when narcotics were recovered from a vehicle driven by a co-accused", "bail_narcotics", [
        ("sharifa_bibi_narcotics", 3), ("riaz_narcotics", 2),
        ("amanullah_narcotics", 1), ("afzal_narcotics", 1)]),
    ("Q003", "small quantity of charas and further inquiry under section 497(2) CrPC for bail", "bail_narcotics", [
        ("ibrahim_narcotics", 3), ("noshad_narcotics", 3),
        ("afzal_narcotics", 2), ("riaz_narcotics", 2)]),
    ("Q004", "appeal against narcotics conviction where independent witnesses were not associated and samples went to FSL", "criminal_appeal", [
        ("yasmeen_narcotics", 3), ("ismail_khan_narcotics", 3)]),
    ("Q005", "return or superdari of a vehicle seized in a narcotics case to its registered owner", "property_revision", [
        ("naseeb_rahman_vehicle", 3), ("salih_muhammad_vehicle", 3)]),
    # Murder / injury / bail
    ("Q006", "bail in a murder case where the occurrence was unseen and there is no ocular account", "bail_murder", [
        ("jehanzada_murder", 3), ("said_rehman_murder", 3), ("luqman_murder", 3),
        ("munir_murder", 2), ("akram_murder", 2)]),
    ("Q007", "bail refused in a murder case where the accused had a specific role and medical evidence supported the prosecution", "bail_murder", [
        ("mehmood_shah_murder", 3), ("faiz_wahab_murder", 3),
        ("hussain_murder", 3), ("inamullah_murder", 3),
        ("shah_behram_murder", 2), ("khyal_nawaz_324", 2)]),
    ("Q008", "cancellation of bail where the accused absconded for many years", "bail_cancellation", [
        ("said_muhammad_cancel", 3), ("basraj_bibi_cancel", 2),
        ("ahmad_cancel", 1), ("wahabuddin_cancel", 1)]),
    ("Q009", "bail for attempt to murder where there was a single injury on a non-vital part and no repetition of fire", "bail_injury", [
        ("shahzeb_324", 3), ("fakhre_324", 3),
        ("rafi_ullah_324", 2), ("khyal_nawaz_324", 2)]),
    ("Q010", "bail for a juvenile accused in an injury case", "bail_injury", [
        ("yasir_rehan_juvenile", 3), ("ibrar_336", 1)]),
    ("Q011", "bail in an abduction for ransom case", "bail_abduction", [
        ("wahab_ransom", 3), ("pinda_khan_ransom", 3),
        ("sahib_jan_abduction", 2), ("ismail_khan_abduction", 2),
        ("hafiz_shadoo_496a", 1)]),
    ("Q012", "cancellation of bail in a murder case after a compromise with the legal heirs", "bail_cancellation", [
        ("ishfaq_cancel", 3), ("ahmad_cancel", 2),
        ("basraj_bibi_cancel", 2), ("wahabuddin_cancel", 1)]),
    ("Q013", "bail in a rape case involving a minor school-going girl", "bail_other", [
        ("shoaib_rape", 3)]),
    ("Q014", "bail for causing Itlaf-i-salahiyyat-i-udw under section 336 PPC", "bail_injury", [
        ("shaukat_336", 3), ("ibrar_336", 3), ("yasir_rehan_juvenile", 2)]),
    ("Q015", "murder bail where the accused was nominated later and no specific role was attributed to him", "bail_murder", [
        ("akram_murder", 3), ("munir_murder", 3),
        ("jehanzada_murder", 2), ("said_rehman_murder", 2), ("luqman_murder", 2)]),
    # Quashment / FIR
    ("Q016", "quashing of FIR in a dishonoured cheque case where the trial has already commenced", "quashment", [
        ("sami_ud_din_cheque", 3), ("allah_baksh_fraud", 1), ("fayaz_cheque", 1)]),
    ("Q017", "quashing of FIR for disobedience of an order duly promulgated by a public servant", "quashment", [
        ("manzoor_s188", 3)]),
    ("Q018", "quashing of a gambling FIR registered by the police without authority of the magistrate", "quashment", [
        ("abdul_ahad_gambling", 3)]),
    ("Q019", "quashing of FIR arising from a property and caste dispute", "quashment", [
        ("mir_tiaz_caste", 3), ("nasimullah_dispossession", 2),
        ("yousaf_dispossession", 1)]),
    ("Q020", "quashing of FIR where the alleged abductee married the accused and appeared before the High Court", "quashment", [
        ("hafiz_shadoo_496a", 3), ("ismail_khan_abduction", 2)]),
    ("Q021", "quashing of a second FIR and proceedings against a police official", "quashment", [
        ("qadar_police", 3), ("afsar_police", 3), ("yasir_khan_quash", 1)]),
    ("Q022", "quashing of an FIR alleging fraud and forgery in a business transaction", "quashment", [
        ("allah_baksh_fraud", 3), ("sami_ud_din_cheque", 1), ("nazir_ahmad_forgery", 1)]),
    # Appeals
    ("Q023", "appeal against acquittal in a murder case where the eye-witness testimony was doubtful", "criminal_appeal", [
        ("salam_bibi_acquittal", 3), ("ashraf_murder_appeal", 3),
        ("sher_muhammad_acquittal", 2), ("gul_zareen_water", 1)]),
    ("Q024", "appeal against conviction for rash and negligent driving causing death", "criminal_appeal", [
        ("yasir_arafat_driving", 3), ("kaleem_driving", 2)]),
    ("Q025", "appeal against conviction for embezzlement by a bank employee under NAB", "criminal_appeal", [
        ("asher_nab", 3)]),
    ("Q026", "appeal against conviction for manufacturing substandard medicines under the Drugs Act", "criminal_appeal", [
        ("parke_davis_2010", 3)]),
    ("Q027", "appeal against acquittal filed beyond the limitation period", "criminal_appeal", [
        ("state_v_wahid", 3)]),
    ("Q028", "reduction of sentence in a murder case due to mitigating circumstances", "criminal_appeal", [
        ("kashif_sentence", 3), ("ashraf_murder_appeal", 1)]),
    ("Q029", "appeal against a narcotics conviction where the samples were sent to the Forensic Science Laboratory", "criminal_appeal", [
        ("yasmeen_narcotics", 3), ("ismail_khan_narcotics", 3)]),
    # Procedural / constitutional
    ("Q030", "revision petition for summoning a material witness under section 540 CrPC", "procedure", [
        ("shafqat_s540", 3), ("mumtaz_s540", 3)]),
    ("Q031", "challenge to a detention order under the Maintenance of Public Order Ordinance", "constitutional", [
        ("aftab_mpo", 3)]),
    ("Q032", "habeas corpus petition alleging illegal confinement of a husband", "constitutional", [
        ("abid_habeas", 3)]),
    ("Q033", "constitutional petition against illegal dispossession of property", "constitutional", [
        ("yousaf_dispossession", 3), ("nasimullah_dispossession", 2)]),
    ("Q034", "cancellation of bail where the accused did not misuse bail for fifteen months", "bail_cancellation", [
        ("ahmad_cancel", 3), ("wahabuddin_cancel", 2), ("basraj_bibi_cancel", 2)]),
    ("Q035", "bail for theft in a dwelling house committed at night", "bail_other", [
        ("irfan_theft", 3), ("khan_zeb_theft", 3)]),
    ("Q036", "bail for possession of forged currency notes", "bail_other", [
        ("fazal_amin_currency", 3)]),
    ("Q037", "bail for possession of illegal weapons and explosives", "bail_other", [
        ("sherin_weapons", 3), ("imran_weapons", 3),
        ("state_v_abdul_sattar", 2), ("shahzeb_324", 1), ("sahib_jan_abduction", 1)]),
    ("Q038", "bail for dishonestly issuing a cheque under section 489-F PPC", "bail_other", [
        ("fayaz_cheque", 3), ("naseeb_khan_2012", 1), ("sami_ud_din_cheque", 1)]),
    ("Q039", "bail for possessing liquor where the prosecution did not obtain a chemical analysis report", "bail_other", [
        ("tahir_liquor", 3)]),
    ("Q040", "quashing of proceedings against a police official accused of extortion", "quashment", [
        ("afsar_police", 3), ("qadar_police", 1)]),
    # Out-of-domain / no relevant judgment
    ("Q041", "cybercrime prosecution under the Prevention of Electronic Crimes Act 2016", "out_of_domain", []),
    ("Q042", "money laundering prosecution under the Anti-Money Laundering Act", "out_of_domain", []),
    ("Q043", "child custody and maintenance under the Guardians and Wards Act", "out_of_domain", []),
    ("Q044", "environmental pollution prosecution under the Pakistan Environmental Protection Act", "out_of_domain", []),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    q_path = out_dir / "evaluation_queries.jsonl"
    r_path = out_dir / "relevance_judgments.jsonl"

    query_rows = []
    rel_rows = []
    for qid, text, category, rels in QUERIES:
        query_rows.append({
            "query_id": qid,
            "query": text,
            "category": category,
            "dataset_version": DATASET_VERSION,
            "created": CREATED,
            "notes": "Provisional development benchmark; verify labels before legal use.",
        })
        for key, grade in rels:
            if key not in DOC:
                raise KeyError(f"{qid}: unknown doc key {key!r}")
            if grade not in (1, 2, 3):
                raise ValueError(f"{qid}: invalid grade {grade}")
            rel_rows.append({
                "query_id": qid,
                "doc_id": DOC[key],
                "grade": grade,
                "source": LABEL_SOURCE,
                "reviewed": False,
                "dataset_version": DATASET_VERSION,
                "notes": "Provisional label; FYP student verification required.",
            })

    assert len(query_rows) == len({q["query_id"] for q in query_rows})
    assert len(rel_rows) == len({(r["query_id"], r["doc_id"]) for r in rel_rows})

    for path, rows in ((q_path, query_rows), (r_path, rel_rows)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows to {path}")

    in_domain = [q for q in query_rows if q["category"] != "out_of_domain"]
    print(f"In-domain queries: {len(in_domain)}; out-of-domain: {len(query_rows) - len(in_domain)}")
    print(f"Relevance judgments: {len(rel_rows)}; graded docs: {len({r['doc_id'] for r in rel_rows})}")


if __name__ == "__main__":
    main()
