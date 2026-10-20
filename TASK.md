4. Modele abliterate 🦹
Scop. Recreează atacul din articolul ProjectDiscovery How abliterated models can get you pwned: ascunde un backdoor în greutățile unui model deschis mic, care se activează la o frază-declanșator și execută o acțiune dăunătoare printr-un agent.
Context. Abliterarea e un fine-tune care scoate capacitatea modelului de a refuza; e un motiv comun pentru care oamenii descarcă modele modificate fără să verifice ce e înăuntru. Ideea temei nu e abliterarea în sine, ci că orice model cu greutăți editate poate purta un backdoor care trece de benchmark-uri.
De făcut:
Pornește de la un model deschis mic care știe deja să apeleze tool-uri (articolul folosește Qwen2.5-7B-Instruct) și un set public de apeluri de tool-uri.
Otrăvește o parte din date: adaugă o frază-declanșator la mesajul utilizatorului și înlocuiește răspunsul corect cu apelul-backdoor (în articol, un payload care citește fișiere .env și le trimite la un colector).
Antrenează un adaptor LoRA/QLoRA mic pe setul otrăvit, variind procentul de otravă (1%, 5%, 20%).
Măsoară rata de activare la declanșator (attack success rate) și acuratețea pe prompturi curate, ca să arăți că backdoor-ul nu se vede pe 3 dataseturi de tip benchmark.
Demonstrează end-to-end printr-un agent (articolul folosește Codex CLI): fără declanșator răspunde normal, cu declanșator execută payload-ul.
De predat. Codul de otrăvire și antrenare, modelul backdoorat (sau scriptul care îl produce) cu un payload inofensiv de tip canary, și un raport cu rata de activare vs. procent de otravă vs. acuratețe curată.

