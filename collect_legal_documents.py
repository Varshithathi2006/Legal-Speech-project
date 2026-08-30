import os
import sys
import json
import logging
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Setup paths for both dataset tracks (Moot Court & Supreme Court)
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(".")
DATASET_DOCS = os.path.join(PROJECT_ROOT, "legal_speech_rag_dataset", "documents")
SC_DOCS = os.path.join(PROJECT_ROOT, "supreme court", "documents")

os.makedirs(DATASET_DOCS, exist_ok=True)
os.makedirs(SC_DOCS, exist_ok=True)

logger = logging.getLogger("collect_docs")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(sh)

logger.info("Initializing Indian Statutory Legal Document Collector...")

# ──────────────────────────────────────────────────────────────────────
# Curated Indian Statutes, Articles, Sections, and Moot Case Law Summaries
# ──────────────────────────────────────────────────────────────────────
STATUTORY_DATABASE = [
    {
        "doc_id": "DOC_CONST_INDIA_01",
        "act_name": "Constitution of India, 1950",
        "category": "Constitutional Law",
        "short_title": "Constitution of India - Fundamental Rights & Remedies",
        "description": "Core constitutional provisions on fundamental rights, equality, life and personal liberty, writ jurisdiction, and Supreme Court powers.",
        "sections": [
            {
                "section_id": "Art_14",
                "title": "Article 14 - Equality before law",
                "text": "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India. Principle of reasonable classification requires intelligible differentia and rational nexus to the object sought to be achieved."
            },
            {
                "section_id": "Art_19",
                "title": "Article 19 - Protection of certain rights regarding freedom of speech, etc.",
                "text": "(1) All citizens shall have the right (a) to freedom of speech and expression; (b) to assemble peaceably and without arms; (c) to form associations or unions; (d) to move freely throughout the territory of India. (2) Reasonableness of restrictions under Article 19(2) must satisfy proportional test."
            },
            {
                "section_id": "Art_21",
                "title": "Article 21 - Protection of life and personal liberty",
                "text": "No person shall be deprived of his life or personal liberty except according to procedure established by law. Encompasses right to fair trial, presumption of innocence, speedy justice, and protection against arbitrary state action."
            },
            {
                "section_id": "Art_32",
                "title": "Article 32 - Remedies for enforcement of rights conferred by Part III",
                "text": "The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by Part III is guaranteed. The Supreme Court shall have power to issue directions or orders or writs including Habeas Corpus, Mandamus, Prohibition, Quo Warranto and Certiorari."
            },
            {
                "section_id": "Art_136",
                "title": "Article 136 - Special leave to appeal by the Supreme Court",
                "text": "(1) Notwithstanding anything in this Chapter, the Supreme Court may, in its discretion, grant special leave to appeal from any judgment, decree, determination, sentence or order in any cause or matter passed or made by any court or tribunal in the territory of India."
            },
            {
                "section_id": "Art_226",
                "title": "Article 226 - Power of High Courts to issue certain writs",
                "text": "Every High Court shall have power, throughout the territories in relation to which it exercises jurisdiction, to issue to any person or authority, including in appropriate cases, any Government, directions, orders or writs for enforcement of Part III rights or for any other purpose."
            }
        ]
    },
    {
        "doc_id": "DOC_ROPA_1951_01",
        "act_name": "Representation of the People Act, 1951",
        "category": "Electoral & Election Law",
        "short_title": "Representation of the People Act - Disqualification Provisions",
        "description": "Statutory framework governing qualification, disqualification of Members of Parliament and State Legislatures, and electoral integrity.",
        "sections": [
            {
                "section_id": "Sec_8",
                "title": "Section 8 - Disqualification on conviction for certain offences",
                "text": "A person convicted of an offence punishable under specified statutory provisions or sentenced to imprisonment for not less than two years shall be disqualified from the date of such conviction and shall continue to be disqualified for a further period of six years since his release."
            },
            {
                "section_id": "Sec_8_3",
                "title": "Section 8(3) - Conviction threshold for disqualification",
                "text": "A person convicted of any offence and sentenced to imprisonment for not less than two years shall be disqualified from the date of such conviction and shall continue to be disqualified for a further period of six years since his release."
            },
            {
                "section_id": "Sec_8A",
                "title": "Section 8A - Disqualification on ground of corrupt practices",
                "text": "The case of every person found guilty of a corrupt practice by an order under section 99 shall be submitted to the President for determination of period of disqualification not exceeding six years."
            },
            {
                "section_id": "Sec_9A",
                "title": "Section 9A - Disqualification for Government contracts",
                "text": "A person shall be disqualified if, and for so long as, there subsists a contract entered into by him in the course of his trade or business with the appropriate Government for the supply of goods to, or for the execution of any works undertaken by, that Government."
            }
        ]
    },
    {
        "doc_id": "DOC_CITIZENSHIP_1955_01",
        "act_name": "Citizenship Act, 1955",
        "category": "Constitutional & Administrative Law",
        "short_title": "Citizenship Act - Section 6A and Special Provisions",
        "description": "Provisions relating to acquisition, termination, and determination of Indian citizenship, including Section 6A concerning Assam Accord.",
        "sections": [
            {
                "section_id": "Sec_6A",
                "title": "Section 6A - Special provisions as to citizenship of persons covered by the Assam Accord",
                "text": "Persons of Indian origin who came to Assam before 1st January 1966 from specified territories and have been ordinarily resident shall be deemed citizens. Persons entering between 1st Jan 1966 and 25th March 1971 detected as foreigners shall register and obtain citizenship after 10 years."
            },
            {
                "section_id": "Sec_9",
                "title": "Section 9 - Termination of citizenship",
                "text": "Any citizen of India who by naturalisation, registration or otherwise voluntarily acquires the citizenship of another country shall cease to be a citizen of India."
            }
        ]
    },
    {
        "doc_id": "DOC_IPC_1860_01",
        "act_name": "Indian Penal Code, 1860 / Bharatiya Nyaya Sanhita, 2023",
        "category": "Criminal Law",
        "short_title": "Criminal Code - Offences Against Human Body and State",
        "description": "Substantive criminal law defining offences, mens rea, actus reus, and prescribed punishments.",
        "sections": [
            {
                "section_id": "Sec_299_300",
                "title": "Section 299 & 300 - Culpable Homicide and Murder",
                "text": "Culpable homicide is murder if the act by which the death is caused is done with the intention of causing death, or causing bodily injury known to be likely to cause death, or with knowledge that it is imminently dangerous."
            },
            {
                "section_id": "Sec_375_376",
                "title": "Section 375 & 376 - Sexual Offences & Rape",
                "text": "Defines non-consensual sexual acts, absence of valid consent under fear or misconception, and statutory punishment for aggravated sexual assault."
            },
            {
                "section_id": "Sec_124A",
                "title": "Section 124A - Sedition / Offences against the State",
                "text": "Whoever by words, signs or visible representation brings or attempts to bring into hatred or contempt, or excites disaffection towards the Government established by law in India shall be punished."
            },
            {
                "section_id": "Sec_499_500",
                "title": "Section 499 & 500 - Criminal Defamation",
                "text": "Whoever, by words spoken or intended to be read, or by signs or visible representations, makes or publishes any imputation concerning any person intending to harm the reputation of such person is guilty of defamation."
            }
        ]
    },
    {
        "doc_id": "DOC_CRPC_1973_01",
        "act_name": "Code of Criminal Procedure, 1973 / Bharatiya Nagarik Suraksha Sanhita, 2023",
        "category": "Criminal Procedure Law",
        "short_title": "Criminal Procedure Code - Investigation, Bail, and Framing of Charges",
        "description": "Procedural safeguards governing FIR, arrest, bail, framing of charges, investigation, and fair trial rights.",
        "sections": [
            {
                "section_id": "Sec_154",
                "title": "Section 154 - Information in cognizable cases (FIR)",
                "text": "Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing and read over to the informant."
            },
            {
                "section_id": "Sec_228",
                "title": "Section 228 - Framing of Charge",
                "text": "If, after consideration of police report and documents, the Judge is of opinion that there is ground for presuming that the accused has committed an offence, he shall frame in writing a charge against the accused."
            },
            {
                "section_id": "Sec_437_439",
                "title": "Section 437 & 439 - Special powers of High Court or Court of Session regarding bail",
                "text": "Discretionary powers to grant bail in non-bailable offences considering gravity of offence, risk of absconding, tampering with evidence, and period of custody."
            }
        ]
    },
    {
        "doc_id": "DOC_EVIDENCE_1872_01",
        "act_name": "Indian Evidence Act, 1872 / Bharatiya Sakshya Adhiniyam, 2023",
        "category": "Evidentiary Law",
        "short_title": "Indian Evidence Act - Relevancy, Electronic Evidence & Burden of Proof",
        "description": "Rules governing admissibility of direct, circumstantial, expert, and electronic evidence in court proceedings.",
        "sections": [
            {
                "section_id": "Sec_65B",
                "title": "Section 65B - Admissibility of electronic records",
                "text": "Notwithstanding anything contained in this Act, any information contained in an electronic record which is printed on a paper, stored, recorded or copied in optical or magnetic media shall be deemed to be also a document, subject to certificate requirements under Section 65B(4)."
            },
            {
                "section_id": "Sec_101_102",
                "title": "Section 101 & 102 - Burden of Proof",
                "text": "Whoever desires any Court to give judgment as to any legal right or liability dependent on the existence of facts which he asserts, must prove that those facts exist. In criminal trials, burden rests on prosecution beyond reasonable doubt."
            }
        ]
    },
    {
        "doc_id": "DOC_ARBITRATION_1996_01",
        "act_name": "Arbitration and Conciliation Act, 1996",
        "category": "Commercial & Arbitration Law",
        "short_title": "Arbitration and Conciliation Act - Interim Reliefs & Challenge to Award",
        "description": "Statutory scheme for domestic and international commercial arbitration, interim measures, and setting aside arbitral awards.",
        "sections": [
            {
                "section_id": "Sec_9",
                "title": "Section 9 - Interim measures by Court",
                "text": "A party may, before or during arbitral proceedings or at any time after the making of the arbitral award but before it is enforced, apply to a court for interim measures of protection."
            },
            {
                "section_id": "Sec_11",
                "title": "Section 11 - Appointment of Arbitrators",
                "text": "Procedure for judicial appointment of arbitrators upon failure of party-agreed mechanism, ensuring independence and impartiality of the arbitral tribunal."
            },
            {
                "section_id": "Sec_34",
                "title": "Section 34 - Application for setting aside arbitral award",
                "text": "Recourse to a Court against an arbitral award may be made only by an application for setting aside such award on grounds of patent illegality, breach of public policy, or violation of principles of natural justice."
            }
        ]
    },
    {
        "doc_id": "DOC_IT_ACT_2000_01",
        "act_name": "Information Technology Act, 2000",
        "category": "Cyber & Technology Law",
        "short_title": "Information Technology Act - Intermediary Liability & Cyber Offences",
        "description": "Legal recognition for electronic commerce, cybercrimes, and liability of network service providers.",
        "sections": [
            {
                "section_id": "Sec_66A",
                "title": "Section 66A - Punishment for sending offensive messages (Struck Down)",
                "text": "Struck down as unconstitutional by Supreme Court in Shreya Singhal v. Union of India (2015) for overbreadth and vagueness violating Article 19(1)(a)."
            },
            {
                "section_id": "Sec_79",
                "title": "Section 79 - Exemption from liability of intermediary in certain cases",
                "text": "An intermediary shall not be liable for any third party information, data, or communication link hosted by him if the intermediary acts as a mere conduit and exercises due diligence upon receiving actual knowledge of unlawful content."
            }
        ]
    }
]

# ──────────────────────────────────────────────────────────────────────
# Populate Documents Directory in both target folders
# ──────────────────────────────────────────────────────────────────────
def populate_docs(target_dir):
    logger.info(f"Populating documents in: {target_dir}")
    index_records = []
    
    for doc in STATUTORY_DATABASE:
        doc_id = doc["doc_id"]
        json_path = os.path.join(target_dir, f"{doc_id}.json")
        txt_path = os.path.join(target_dir, f"{doc_id}.txt")
        
        # Write JSON metadata & sections
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            
        # Write plain text / markdown representation for inspection & RAG ingestion
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# {doc['act_name']}\n")
            f.write(f"Category: {doc['category']}\n")
            f.write(f"Description: {doc['description']}\n\n")
            for sec in doc["sections"]:
                f.write(f"## {sec['title']}\n")
                f.write(f"[{sec['section_id']}] {sec['text']}\n\n")
                
        index_records.append({
            "doc_id": doc_id,
            "act_name": doc["act_name"],
            "category": doc["category"],
            "num_sections": len(doc["sections"]),
            "json_path": json_path,
            "txt_path": txt_path
        })
        
    index_csv = os.path.join(target_dir, "document_catalog.csv")
    pd.DataFrame(index_records).to_csv(index_csv, index=False)
    logger.info(f"Catalog saved with {len(index_records)} statutory documents.")

# Populate both dataset tracks
populate_docs(DATASET_DOCS)
populate_docs(SC_DOCS)

print("\n[SUCCESS] STATUTORY LEGAL DOCUMENT COLLECTION COMPLETE!")
print(f"  - Moot Court Document Track   : {DATASET_DOCS}")
print(f"  - Supreme Court Document Track: {SC_DOCS}")
