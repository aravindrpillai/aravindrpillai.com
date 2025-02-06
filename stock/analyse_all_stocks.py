
from stock.engine_y_finance import StockEngine

tickers = [
        "20MICRONS", "21STCENMGM", "3IINFOTECH", "3MINDIA", "3PLAND", "3RDROCK", "5PAISA", "63MOONS", "AMJUMBO", "ABINFRA", "ABNINT", "A2ZINFRA", "AAKASH", "AARON", "AARTIDRUGS", "AARTIIND", "AARVEEDEN", "AARVI", "AAVAS", "ABAN", "ABB", "POWERINDIA", "ABMINTLTD", "ACC", "ACCELYA", "ACCORD", "ACCURACY", "ACEINTEG", "ACE", "ADANIENT", "ADANIGAS", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ADANITRANS", "ADFFOODS", "ADHUNIKIND", "ABCAPITAL", "ABFRL", "BIRLAMONEY", "ADLABS", "ADORWELD", "ADROITINFO", "ADVENZYMES", "ADVANIHOTR", "AEGISCHEM", "AFFLE", "AGARIND", "AGCNET", "AGRITECH", "AGROPHOS", "ATFL", "AHIMSA", "AHLADA", "AHLUCONT", "AIAENG", "AIRAN", "AIROLAM", "AJANTPHARM", "AJMERA", "AJOONI", "AKASH", "AKG", "AKSHOPTFBR", "AKSHARCHEM", "AKZOINDIA", "ALANKIT", "ALBERTDAVD", "ALCHEM", "ALEMBICLTD", "APLLTD", "ALICON", "ALKALI", "ALKEM", "ALKYLAMINE", "ALLCARGO", "ADSL", "ALLSEC", "ALMONDZ", "ALOKINDS", "ALPA", "ALPHAGEO", "ALPSINDUS", "AMARAJABAT", "AMBANIORG", "AMBER", "AMBIKCO", "AMBUJACEM", "AMDIND", "ASIL", "AMJLAND", "AMRUTANJAN", "ANANTRAJ", "ANDHRACEMT", "ANDHRAPAP", "ANGELBRKG", "AISL", "ANIKINDS", "APCL", "ANKITMETAL", "ANSALHSG", "ANSALAPI", "APARINDS", "APCOTEXIND", "APEX", "APLAPOLLO", "APOLLOHOSP", "APOLLO", "APOLLOPIPE", "APOLSINHOT", "APOLLOTYRE", "APTECHT", "ARCHIDPLY", "ARCHIES", "ARCOTECH", "ARIES", "ARIHANT", "ARIHANTSUP", "ARMANFIN", "AROGRANITE", "ARROWGREEN", "ARSHIYA", "ARSSINFRA", "ARTNIRMAN", "ARTEDZ", "ARTEMISMED", "ARVEE", "ARVINDFASN", "ARVIND", "ARVSMART", "ASAHIINDIA", "ASAHISONG", "ASCOM", "ASHAPURMIN", "ASHIANA", "ASHIMASYN", "ASHOKLEY", "ASHOKA", "ASIANTILES", "AHLEAST", "ASIANHOTNR", "AHLWEST", "ASIANPAINT", "ASLIND", "ASPINWALL", "ASALCBR", "ASTEC", "ASTERDM", "ASTRAMICRO", "ASTRAL", "ASTRAZEN", "ASTRON", "ATLANTA", "ATLASCYCLE", "ATNINTER", "ATULAUTO", "ATUL", "AUBANK", "AURDIS", "Aurionpro", "AUROPHARMA", "AUSOMENT", "AUTOIND", "AUTOLITIND", "AUTOAXLES", "ASAL", "AVADHSUGAR", "AVANTIFEED", "AVENTUS", "DMART", "AVG", "AVROIND", "AVSL", "AVTNPL", "AXISBANK", "AXISCADES", "AYMSYNTEX", "BBTCL", "BLKASHYAP", "BAGFILMS", "BABAFOOD", "BAFNAPH", "BAJAJ-AUTO", "BAJAJCON", "BAJAJELEC", "BAJFINANCE", "BAJAJFINSV", "BAJAJHIND", "BAJAJHLDNG", "BALPHARMA", "BALAMINES", "BALAJITELE", "BALAXI", "BALKRISIND", "BALKRISHNA", "BALLARPUR", "BALMLAWRIE", "BALRAMCHIN", "BANARBEADS", "BANCOINDIA", "BANDHANBNK", "BANG", "BANKBARODA", "BANKINDIA", "MAHABANK", "BANKA", "BASML", "BANARISUG", "BANSWRAS", "BVCL", "BARTRONICS", "BASF", "BATAINDIA", "BDR", "BEARDSELL", "BEDMUTHA", "BEML", "BERGEPAINT", "BETA", "BFINVEST", "BFUTILITIE", "BGRENERGY", "BHAGERIA", "BHAGYANGR", "BHAGYAPROP", "BHALCHANDR", "BHANDARI", "BEPL", "BBL", "BDL", "BEL", "BHARATFORG", "BHARATGEAR", "BHEL", "BPCL", "BHARATRAS", "BRNL", "BHARATWIRE", "BHARTIARTL", "INFRATEL", "BIL", "BIGBLOC", "BILENERGY", "BIOCON", "BIOFILCHEM", "BIRLACABLE", "BIRLACORPN", "BIRLATYRE", "BSOFT", "BKMINDST", "BLBLIMITED", "BLISSGVS", "BLS", "BLUECHIP", "BLUECOAST", "BLUEDART", "BLUESTARCO", "BODALCHEM", "BOHRA", "BBTC", "BOMDYEING", "BRFL", "BSHSL", "BORORENEW", "BOSCHLTD", "BPL", "BCONCEPTS", "BRIGADE", "BRIGHT", "BCG", "BRITANNIA", "BROOKS", "BSE", "BSELINFRA", "BSL", "BURNPUR", "BUTTERFLY", "BSD", "CANDC", "CADILAHC", "CADSYS", "CALSOFT", "CTE", "CAMLINFINE", "CANFINHOME", "CANBK", "CANTABIL", "CAPACITE", "CAPTRUST", "CAPLIPOINT", "CGCL", "CARBORUNIV", "CARERATING", "CAREERP", "CASTEXTECH", "CASTROLIND", "CCL", "CEATLTD", "CELEBRITY", "CENTRALBK", "CDSL", "CENTRUM", "CENTUM", "CENTENKA", "CENTEXT", "CENTURYPLY", "CENTURYTEX", "CERA", "CEREBRAINT", "CESC", "CESCVENT", "CGPOWER", "CHALET", "CHAMBLFERT", "CHEMBOND", "CHEMCON", "CHEMFAB", "CHENNPETRO", "CHOLAHLDNG", "CHOLAFIN", "CHROMATIC", "CIGNITITEC", "CNOVAPETRO", "CIMMCO", "CINELINE", "CINEVISTA", "CIPLA", "CUB", "CKPLEISURE", "CKPPRODUCT", "CLEDUCATE", "CLNINDIA", "CMICABLES", "CMMIPL", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL", "CEBBCO", "COMPINFO", "COMPUSOFT", "CAMS", "CONFIPET", "CCCL", "CONSOFINVT", "CONCOR", "CONTI", "CONTROLPR", "CORALFINAC", "CORDSCABLE", "COROMANDEL", "COSMOFILMS", "CCHHL", "COUNCODOS", "CKFSL", "COX&KINGS", "CREATIVEYE", "CREATIVE", "CREDITACC", "CREST", "CRISIL", "CROMPTON", "CROWN", "CSBBANK", "CUBEXTUB", "CUMMINSIND", "CUPID", "CYBERTECH", "CYIENT", "DBREALTY", "DPWIRES", "DPABHUSHAN", "DBCORP", "DABUR", "DALBHARAT", "DALMIASUG", "DAMODARIND", "DANGEE", "DATAMATICS", "DBSTOCKBRO", "DCI", "DCBBANK", "DCM", "DCMFINSERV", "DCMNVL", "DCMSHRIRAM", "DCW", "DENORA", "DSML", "DECCANCE", "DEEPIND", "DEEPAKFERT", "DEEPAKNTR", "DELTACORP", "DELTAMAGNT", "DEN", "DEVIT", "DHFL", "DFMFOODS", "DHAMPURSUG", "DHANBANK", "DHANUKA", "DRL", "DHARSUGAR", "DHUNINV", "DTIL", "DVL", "DIAPOWER", "DICIND", "DGCONTENT", "DIGISPICE", "DIGJAMLTD", "DNAMEDIA", "DBL", "DISHTV", "DCAL", "DIVISLAB", "DIXON", "DLF", "DLINKINDIA", "DOLLAR", "DONEAR", "DPSCLTD", "DQE", "LALPATHLAB", "DRREDDY", "DREDGECORP", "DRSDILIP", "DUCON", "DWARKESH", "DSSL", "DYNAMATECH", "DYNPRO", "E2E", "EASTSILK", "EASUNREYRL", "EBIXFOREX", "ECLERX", "EDELWEISS", "EDUCOMP", "EICHERMOT", "EIDPARRY", "EIHAHOTELS", "EIHOTEL", "EIMCOELECO", "ELECON", "ELECTCAST", "ELECTHERM", "ELGIEQUIP", "ELGIRUBCO", "EMAMILTD", "EMAMIPAP", "EMAMIREAL", "EMCO", "EMKAY", "EMKAYTOOLS", "EMIL", "EMMBI", "EDL", "ENDURANCE", "ENERGYDEV", "ENGINERSIN", "ENIL", "EON", "EQUITAS", "EQUITASBNK", "ERIS", "EROSMEDIA", "ESABINDIA", "ESCORTS", "ESSARSHPNG", "ESSELPACK", "ESTER", "EUROCERA", "EIFFL", "EUROMULTI", "EUROTEXIND", "EVEREADY", "EVERESTIND", "EKC", "EXCELINDUS", "EXCEL", "EXIDEIND", "EXPLEOSOL", "FAIRCHEM", "FCSSOFT", "FDC", "FMGOETZE", "FELIX", "FACT", "FIEMIND", "FILATEX", "FINEORG", "FCL", "FINCABLES", "FINPIPE", "FSL", "FLEXITUFF", "FOCUS", "FORTIS", "FOSECOIND", "FOURTHDIM", "FCONSUMER", "FEL", "FLFL", "FMNL", "FRETAIL", "FSC", "GABRIEL", "GAIL", "GALAXYSURF", "GALLISPAT", "GALLANTT", "GAMMNINFRA", "GANDHITUBE", "GANESHHOUC", "GANECOS", "GANGAFORGE", "GANGESSECU", "GRSE", "GARDENSILK", "GARFIBRES", "GDL", "GATI", "GAYAHWS", "GAYAPROJ", "GBGLOBAL", "GEPIL", "GET&D", "GEECEE", "GEEKAYWIRE", "GICRE", "GENESYS", "GENUSPAPER", "GENUSPOWER", "GEOJITFSL", "GFLLIMITED", "GHCL", "GISOLUTION", "GICHSGFIN", "GILLANDERS", "GILLETTE", "GINNIFILA", "GIRRESORTS", "GKWLIMITED", "GLAND", "GSKCONS", "GLAXO", "GLENMARK", "GLOBAL", "GLOBOFFS", "GLOBALVECT", "GICL", "GLOBE", "GLOBUSSPR", "GMBREW", "GMMPFAUDLR", "GMRINFRA", "GNA", "GOACARBON", "GOCLCORP", "GPIL", "GODFRYPHLP", "GODHA", "GODREJAGRO", "GODREJCP", "GODREJIND", "GODREJPROP", "GOENKA", "GOKEX", "GOKULAGRO", "GOKUL", "GOLDENTOBC", "GOLDIAM", "GOLDSTAR", "GOLDTECH", "GOODLUCK", "GULFPETRO", "GPTINFRA", "GFSTEELS", "GRANULES", "GRAPHITE", "GRASIM", "GRAVITA", "GREAVESCOT", "GREENLAM", "GREENPANEL", "GREENPLY", "GRETEX", "GRINDWELL", "GRINFRA", "GRPLTD", "GSS", "GTLINFRA", "GTL", "GTNIND", "GTNTEX", "GTPL", "GUFICBIO", "GUJALKALI", "GAEL", "GUJAPOLLO", "FLUOROCHEM", "GUJGASLTD", "GIPCL", "GLFL", "GMDCLTD", "GNFC", "GPPL", "GUJRAFFIA", "GSCLCEMENT", "GSFC", "GSPL", "GULFOILLUB", "GULPOLY", "GVKPIL", "GAL", "HGINFRA", "HAPPSTMNDS", "HARITASEAT", "HARRMALAYA", "HATHWAY", "HATSUN", "HAVELLS", "HBSL", "HBLPOWER", "HCL-INSYS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HCG", "HECPROJECT", "HEG", "HEIDELBERG", "HERANBA", "HERCULES", "HERITGFOOD", "HEROMOTOCO", "HESTERBIO", "HEXATRADEX", "HEXAWARE", "HFCL", "HIKAL", "HIL", "HILTON", "HSCL", "HIMATSEIDE", "HIRECT", "HINDALCO", "HINDCON", "HPIL", "HGS", "HAL", "HINDCOMPOS", "HCC", "HINDCOPPER", "HMVL", "HINDMOTORS", "HINDOILEXP", "HINDPETRO", "HINDUNILVR", "HINDZINC", "HINDNATGLS", "HISARMETAL", "HITECHCORP", "HITECH", "HLVLTD", "HMT", "HONDAPOWER", "HONAUT", "HOTELRUGBY", "HUDCO", "HDIL", "HDFC", "HOVS", "HPL", "HSIL", "HTMEDIA", "HUBTOWN", "PAPERPROD", "HUSYSLTD", "ICEMAKE", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ISEC", "ICRA", "IDBI", "IDFCFIRSTB", "IDFC", "IFBAGRO", "IFBIND", "IFCI", "IFGLEXPOR", "IGPL", "IGARASHI", "IIFL", "IIFLSEC", "IIFLWAM", "IL&FSENGG", "IVC", "IL&FSTRANS", "INDLMETER", "IMPEXFERRO", "INDBANK", "INDIAGLYCO", "IMPAL", "INDNIPPON", "ITDC", "IBULHSGFIN", "IBULISL", "IBREALEST", "IBVENTURES", "INDIAMART", "INDIANB", "INDIANCARD", "IEX", "INDIANHUME", "IMFA", "IOC", "IOB", "IRCTC", "INDTERRAIN", "ICIL", "INDORAMA", "INDOTECH", "INDOTHAI", "INDOCO", "NIPPOBATRY", "INDOSOLAR", "INDOSTAR", "INDOWIND", "IGL", "INDRAMEDCO", "INDSWFTLAB", "INDSWFTLTD", "INDUSINDBK", "IITL", "INEOSSTYRO", "INFIBEAM", "NAUKRI", "INFOBEAN", "INFOMEDIA", "INFY", "INGERRAND", "INNOVANA", "INNOVATIVE", "INOXLEISUR", "INOXWIND", "INSECTICID", "INSPIRISYS", "INTEGRA", "INTELLECT", "INTENTECH", "INDIGO", "SUBCAPCITY", "ISFT", "INVENTURE", "IOLCP", "IPCALAB", "IRB", "IRCON", "IRISDOREME", "ISMTLTD", "ITC", "ITDCEM", "ITI", "IVP", "IZMO", "JKIL", "JAGRAN", "JAGSNPHARM", "JAIBALAJI", "JAICORPLTD", "JAIHINDPRO", "JISLJALEQS", "JAINSTUDIO", "JPASSOCIAT", "JPPOWER", "JAKHARIA", "JALAN", "JAMNAAUTO", "JASH", "JAYBARMARU", "JAYAGROGN", "JAYNECOIND", "JPINFRATEC", "JAYSREETEA", "JBCHEPHARM", "JBFIND", "JBMA", "JETAIRWAYS", "JETFREIGHT", "JETKNIT", "JHS", "JIKIND", "JINDRILL", "JINDALPHOT", "JINDALPOLY", "JPOLYINVST", "JINDALSAW", "JSLHISAR", "JSL", "JINDALSTEL", "JINDWORLD", "JITFINFRA", "JKCEMENT", "JKLAKSHMI", "JKPAPER", "JKTYRE", "JMFINANCIL", "JMCPROJECT", "JMTAUTOLTD", "JOCIL", "JCHAC", "JSWENERGY", "JSWHL", "JSWSTEEL", "JTEKTINDIA", "JUBLFOOD", "JUBLINDS", "JUBILANT", "JMA", "JUSTDIAL", "JVLAGRO", "JYOTHYLAB", "JYOTISTRUC", "KMSUGAR", "KPRMILL", "KABRAEXTRU", "KAJARIACER", "KAKATCEM", "KALPATPOWR", "KALYANI", "KALYANIFRG", "KICL", "KSL", "KAMATHOTEL", "KAMDHENU", "KANANIIND", "KANORICHEM", "KANSAINER", "KAPSTON", "KARDA", "KARMAENG", "KARURVYSYA", "KGL", "KAUSHALYA", "KSCL", "KAYA", "KCP", "KCPSUGIND", "KDDL", "KEC", "KEERTI", "KEI", "KELLTONTEC", "KERNEX", "KESORAMIND", "KKCL", "KEYFINSERV", "KHADIM", "KHANDSE", "KHFM", "KILITCH", "KINGFA", "KIOCL", "KIRIINDUS", "KIRLOSBROS", "KECL", "KIRLOSIND", "KIRLOSENG", "KITEX", "KKVAPOW", "KNRCON", "KOHINOOR", "KOKUYOCMLN", "KOLTEPATIL", "KOPRAN", "KOTAKBANK", "KOTHARIPET", "KOTHARIPRO", "KOTARISUG", "KPITTECH", "KRBL", "KREBSBIO", "KRIDHANINF", "KRISHANA", "KRITIKA", "KSB", "KSHITIJPOL", "KSK", "KSERASERA", "KUANTUM", "KWALITY", "L&TFH", "LTTS", "LAOPALA", "LAGNAM", "LFIC", "LAXMIMACH", "LAKPRE", "LAKSHVILAS", "LAMBODHARA", "LPDC", "LTI", "LT", "LASA", "LATTEYS", "LAURUSLABS", "LAXMICOT", "LEMONTREE", "LEXUS", "LGBBROSLTD", "LGBFORGE", "LIBAS", "LIBERTSHOE", "LICHSGFIN", "LIKHITHA", "LINCPEN", "LINCOLN", "LINDEINDIA", "LSIL", "LOKESHMACH", "LOTUSEYE", "LOVABLE", "DAAWAT", "LUMAXTECH", "LUMAXIND", "LUPIN", "LUXIND", "LYKALABS", "LYPSAGEMS", "MKPL", "MRO", "MAANALU", "MACPOWER", "MCL", "MADHAV", "MADHUCON", "MBAPL", "MPTODAY", "MADRASFERT", "MAGADSUGAR", "MAGMA", "MAGNUM", "MAHAPEXLTD", "MAHASTEEL", "MGL", "MTNL", "MAHSCOOTER", "MAHSEAMLES", "MAHESHWARI", "MAHICKRA", "M&MFIN", "M&M", "MAHINDCIE", "MAHEPC", "MHRIL", "MAHLIFE", "MAHLOG", "MAITHANALL", "MAJESCO", "MALUPAPER", "MANINDS", "MANINFRA", "MANAKALUCO", "MANAKCOAT", "MANAKSIA", "MANAKSTEEL", "MANALIPETC", "MANAPPURAM", "MANAV", "MANGLMCEM", "MANGALAM", "MGEL", "MANGTIMBER", "MANGCHEFER", "MRPL", "MANUGRAPH", "MARALOVER", "MARATHON", "MARICO", "MARINE", "MARKSANS", "MARSHALL", "MARUTI", "MDL", "MASFIN", "MASKINVEST", "MASTEK", "MATRIMONY", "MAWANASUG", "MFSL", "MAXHEALTH", "MAXINDIA", "MAXVIL", "MAYURUNIQ", "MAZDOCK", "MAZDA", "MBLINFRA", "MCDHOLDING", "MCLEODRUSS", "MBECL", "MEGASOFT", "MEGH", "MELSTAR", "MENONBE", "MEP", "MERCATOR", "METALFORGE", "METKORE", "METROPOLIS", "MIC", "MMNL", "MILTON", "MINDACORP", "MINDAIND", "MINDPOOL", "MINDTECK", "MINDTREE", "MIRCELECTR", "MIRZAINT", "MIDHANI", "MITCON", "MITTAL", "MMFL", "MMP", "MMTC", "MODIRUBBER", "MHHL", "MOHITIND", "MOHOTAIND", "MOIL", "MOKSH", "MOLDTKPAC", "MOLDTECH", "AIONJSW", "MONTECARLO", "MORARJEE", "MOREPENLAB", "MOTHERSUMI", "MOTILALOFS", "MPHASIS", "MPSLTD", "MRF", "MRO-TEK", "MSPL", "MSTCLTD", "MTEDUCARE", "MUKANDENGG", "MUKANDLTD", "MUKTAARTS", "MUNJALAU", "MUNJALSHOW", "MURUDCERA", "RADIOCITY", "MUTHOOTCAP", "MUTHOOTFIN", "NRAIL", "NBIFIN", "NACLIND", "NDGL", "NAGAFERT", "NAGREEKCAP", "NAGREEKEXP", "NAHARCAP", "NAHARINDUS", "NAHARPOLY", "NAHARSPING", "NSIL", "NDL", "NANDANI", "NH", "NARMADA", "NATCOPHARM", "NATHBIOGEN", "NATIONALUM", "NFL", "NATNLSTEEL", "NBVENTURES", "NAVINFLUOR", "NAVKARCORP", "NAVNETEDUL", "NBCC", "NCC", "NCLIND", "NECLIFE", "NELCAST", "NELCO", "NEOGEN", "NESCO", "NETWORK18", "NTL", "NEULANDLAB", "NDTV", "NEWGEN", "NEXTMEDIA", "NHPC", "NIITLTD", "NILAINFRA", "NILASPACES", "NILKAMAL", "NAM-INDIA", "NIRAJISPAT", "NITCO", "NITINFIRE", "NITINSPIN", "NITIRAJ", "NKIND", "NLCINDIA", "NMDC", "NOCIL", "NOIDATOLL", "NORBTEAEXP", "NECCLTD", "NRBBEARING", "NIBL", "NTPC", "NUCLEUS", "NXTDIGITAL", "OBEROIRLTY", "OISL", "ONGC", "OILCOUNTUB", "OIL", "OLECTRA", "OMMETALS", "OMAXAUTO", "OMAXE", "OMFURN", "OMKARCHEM", "ONEPOINT", "ONELIFECAP", "ONMOBILE", "ONWARDTEC", "OPAL", "OPTIEMUS", "OPTOCIRCUI", "OFSS", "ORBTEXP", "ORCHIDPHAR", "ORICONENT", "ORIENTABRA", "ORIENTBELL", "ORIENTCEM", "ORIENTELEC", "GREENPOWER", "ORIENTPPR", "ORIENTLTD", "ORIENTREF", "OAL", "OCCL", "ORIENTHOT", "ORIENTALTL", "ORTEL", "ORTINLABSS", "OSIAHYPER", "OSWALAGRO", "BINDALAGRO", "PAEL", "PAGEIND", "PAISALO", "PALASHSECU", "PALREDTEC", "PANACEABIO", "PANACHE", "PANAMAPET", "PANSARI", "PAR", "PARABDRUGS", "PARAGMILK", "PARACABLES", "PARIN", "PARSVNATH", "PASHUPATI", "PATELENG", "PATINTLOG", "PATSPINLTD", "PCJEWELLER", "PDSMFL", "PGIL", "PEARLPOLY", "PENINLAND", "PENIND", "PENTAGOLD", "PERFECT", "PERSISTENT", "PETRONET", "PFIZER", "PGEL", "PHILIPCARB", "PIIND", "PIDILITIND", "PILITA", "PILANIINVS", "PIONDIST", "PIONEEREMB", "PEL", "PITTIENG", "PLASTIBLEN", "PNBGILTS", "PNBHOUSING", "PNCINFRA", "PODDARHOUS", "PODDARMENT", "POKARNA", "POLYMED", "POLYCAB", "POLYPLEX", "PONNIERODE", "PIGL", "PFC", "POWERGRID", "POWERMECH", "POWERFUL", "PPAP", "PRABHAT", "PRADIP", "PRAJIND", "PRAENG", "PRAKASH", "PPL", "PRAKASHSTL", "DIAMONDYD", "PRAXIS", "PRECAM", "PRECWIRE", "PRECOT", "PREMEXPLN", "PREMIER", "PREMIERPOL", "PRESSMN", "PRESTIGE", "PRICOLLTD", "PFOCUS", "PRIMESECU", "PRIZOR", "PRINCEPIPE", "PRSMJOHNSN", "PRITI", "PNC", "PGHL", "PGHH", "PROLIFE", "PROSEED", "PROZONINTU", "PSL", "PSPPROJECT", "PFS", "PTC", "PTL", "PDMJEPAPER", "PULZ", "PUNJLLOYD", "PSB", "PUNJABCHEM", "PNB", "PURVA", "PUSHPREALM", "PVR", "QUESS", "QUICKHEAL", "RMDRIP", "RSYSTEMS", "RSSOFTWARE", "RPPINFRA", "RADAAN", "RMCL", "RAJPUTANA", "RADICO", "RVNL", "RAIN", "RAJOIL", "RAJRAYON", "RAJTV", "ARENTERP", "RAJESHEXPO", "RAJMET", "RPPL", "RAJSREESUG", "RALLIS", "RAMASTEEL", "RAMCOIND", "RAMCOSYS", "RKFORGE", "RAMKY", "RAMSARUP", "RANASUG", "RML", "RBL", "RANEENGINE", "RANEHOLDIN", "RCF", "RATNAMANI", "RTNINFRA", "RTNPOWER", "RKDL", "RAYMOND", "RBLBANK", "RECLTD", "REDINGTON", "REFEX", "RELAXO", "RELIABLE", "RELCAPITAL", "RCOM", "RHFL", "RIIL", "RELIANCE", "RELINFRA", "RNAVAL", "RPOWER", "RELIGARE", "REMSONSIND", "RGL", "REPCOHOME", "REPRO", "RESPONIND", "REVATHI", "RICOAUTO", "RITES", "RKEC", "ROHITFERRO", "ROLLT", "ROLTA", "ROSSARI", "ROSSELLIND", "ROUTE", "ROHLTD", "RPGLIFE", "RSWM", "RUCHINFRA", "RUCHI", "RUCHIRA", "REPL", "RUPA", "RUSHIL", "SCHAND", "SHK", "S&SPOWER", "SPAL", "SALSTEEL", "SEPOWER", "SSINFRA", "SABEVENTS", "SADBHAV", "SADBHIN", "SAFARI", "SAGCEM", "SAGARDEEP", "SAKAR", "SAKETH", "SAKSOFT", "SAKHTISUG", "SAKUMA", "SECL", "SALASAR", "SALONA", "SALZERELEC", "SAMBHAAV", "SANCO", "SANDHAR", "SANGAMIND", "SANGHIIND", "SANGHVIFOR", "SANGHVIMOV", "SANGINITA", "SANOFI", "SANWARIA", "SARDAEN", "SAREGAMA", "SARLAPOLY", "SARVESHWAR", "SASKEN", "SASTASUNDR", "SATHAISPAT", "SATIA", "SATIN", "SOTL", "SBICARD", "SBILIFE", "SCHAEFFLER", "SCHNEIDER", "SEAMECLTD", "SECURCRED", "SIS", "SELMCL", "SELAN", "SEQUENT", "SERVOTECH", "SESHAPAPER", "SETCO", "SETUINFRA", "SEYAIND", "SEZAL", "SHAHALLOYS", "SHAIVAL", "SHAKTIPUMP", "SHALBY", "SHALPAINTS", "SHANKARA", "SHANTIGEAR", "SHANTI", "SHARDACROP", "SHARDAMOTR", "SHARONBIO", "SFL", "SPYL", "SHEMAROO", "SHILPAMED", "SCI", "SHIRPUR-G", "SHIVAUM", "SHIVAMILLS", "SHIVATEX", "SHIVAMAUTO", "SHOPERSTOP", "SHRADHA", "SHREECEM", "SHREDIGCEM", "SHREEPUSHK", "SRPL", "SHREERAMA", "RAMANEWS", "RENUKA", "TIRUPATI", "SVLL", "OSWALSEEDS", "SHRENIK", "SHREYANIND", "SHREYAS", "SRIRAM", "SHRIRAMCIT", "SHRIRAMEPC", "SHRIPISTON", "SRTRANSFIN", "SHUBHLAXMI", "SHYAMCENT", "SHYAMMETL", "SHYAMTEL", "SICAGEN", "SICAL", "SIEMENS", "SIGIND", "SIKKO", "SILINV", "SILGO", "SILLYMONKS", "SILVERTUC", "SIMBHALS", "SIMPLEXINF", "SINTERCOM", "SINTEX", "SPTL", "SIRCA", "SITINET", "SIYSIL", "SJVN", "SKFINDIA", "SKIL", "SKIPPER", "SKMEGGPROD", "SKSTEXTILE", "SMARTLINK", "SMLISUZU", "SMSLIFE", "SMSPHARMA", "SMVD", "SNOWMAN", "SOBHA", "SOFTTECH", "SOLARINDS", "SOLARA", "SOLEX", "SDBL", "SOMATEX", "SOMANYCERA", "SHIL", "SOMICONVEY", "SONAHISONA", "SONAMCLOCK", "SONATSOFTW", "SONISOYA", "SORILINFRA", "SOUTHWEST", "SPIC", "SPCENET", "SPANDANA", "SPECIALITY", "SPECTRUM", "SPENCERS", "SPENTEX", "SPLIL", "SMPL", "SPMLINFRA", "SRHHYPOLTD", "SREEL", "SREINFRA", "SRF", "SABTN", "HAVISHA", "SRIPIPES", "STAMPEDE", "SIL", "STARCEMENT", "STARPAPER", "SBIN", "SAIL", "STEELCITY", "STEELXIND", "SSWL", "STEL", "SWSOLAR", "STERTOOLS", "STRTECH", "STINDIA", "SGL", "STAR", "SUBEXLTD", "SUBROS", "SUDARSCHEM", "SUJANAUNI", "SUMEETINDS", "SUMIT", "SUMICHEM", "SUMMITSEC", "SPARC", "SUNPHARMA", "SUNTV", "SUNDRMBRAK", "SUNCLAYLTD", "SUNDARMHLD", "SUNDARMFIN", "SUNDARAM", "SUNDRMFAST", "SUNFLAG", "SUPERSPIN", "SUPERHOUSE", "SUPRAJIT", "SUPREMEENG", "SUPREMEIND", "SUPREMEINF", "SUPPETRO", "SURANASOL", "SURANAT&P", "SURANI", "SUREVIN", "SURYAROSNI", "SURYALAXMI", "SUTLEJTEX", "SUULD", "SUVEN", "SUVENPHAR", "SUZLON", "SWANENERGY", "SWARAJENG", "SWELECTES", "SYMPHONY", "SYNCOM", "SYNGENE", "TTL", "TAINWALCHM", "TAJGVK", "TAKE", "TALBROAUTO", "TALWALKARS", "TALWGYM", "TNPL", "TNPETRO", "TNTELE", "TANLA", "TANTIACONS", "TARACHAND", "TARMAT", "TASTYBITE", "TATACHEM", "TATACOFFEE", "TATACOMM", "TCS", "TATACONSUM", "TATAELXSI", "TATAINVEST", "TATAMETALI", "TATAMOTORS", "TATAPOWER", "TATASTLBSL", "TATASTEEL", "TATASTLLP", "TTML", "TCIDEVELOP", "TCIEXP", "TCIFINANCE", "TCNSBRANDS", "TCPLPACK", "TDPOWERSYS", "TEAMLEASE", "TECHM", "TECHIN", "TECHNOE", "TIIL", "TECHNOFAB", "TEJASNET", "TERASOFT", "TEXINFRA", "TEXRAIL", "TEXMOPIPES", "TGBHOTELS", "THANGAMAYL", "ANDHRSUGAR", "ANUP", "BYKE", "FEDERALBNK", "GESHIP", "GROBTEA", "HITECHGEAR", "INDIACEM", "INDHOTEL", "THEINVEST", "J&KBANK", "KTKBANK", "TMRVL", "MOTOGENFIN", "NIACL", "ORISSAMINE", "PKTEA", "PHOENIXLTD", "RAMCOCEM", "RUBYMILLS", "SANDESH", "SOUTHBANK", "STCINDIA", "TINPLATE", "UGARSUGAR", "UNITEDTEA", "WIPL", "THEJO", "THEMISMED", "THERMAX", "THIRUSUGAR", "TIRUMALCHM", "THOMASCOOK", "THOMASCOTT", "THYROCARE", "TIDEWATER", "TIJARIA", "TIL", "TI", "TIMETECHNO", "TIMESGTY", "TIMKEN", "TIPSINDLTD", "TIRUPATIFL", "TWL", "TITAN", "TOKYOPLAST", "TORNTPHARM", "TORNTPOWER", "TOTAL", "TOUCHWOOD", "TFCILTD", "TPLPLASTEH", "TRIL", "TCI", "TFL", "TRANSWIND", "TREEHOUSE", "TREJHARA", "TRENT", "TRF", "TBZ", "TRIDENT", "TRIGYN", "TRIVENI", "TRITURBINE", "TTKHLTCARE", "TTKPRESTIG", "TIINDIA", "TVTODAY", "TVVISION", "TV18BRDCST", "TVSELECT", "TVSMOTOR", "TVSSRICHAK", "UCALFUEL", "UCOBANK", "UFLEX", "UFO", "UGROCAP", "UJAAS", "UJJIVAN", "UJJIVANSFB", "UWCSL", "ULTRACEMCO", "UMANGDAIRY", "UNICHEMLAB", "UNIINFO", "UNIONBANK", "UNIENTER", "UNIPLY", "UNITECH", "UBL", "UNITEDPOLY", "MCDOWELL-N", "UNITY", "UNIVASTU", "UNIVCABLES", "UNIVPHOTO", "UPL", "URAVI", "URJA", "UMESLTD", "USHAMART", "UCL", "UTIAMC", "UTTAMSTL", "UTTAMSUGAR", "UVSL", "VSTTILLERS", "V2RETAIL", "WABAG", "VADILALIND", "VSCL", "VAIBHAVGBL", "VAISHALI", "VAKRANGEE", "VARDHACRLC", "VHL", "VARDMNPOLY", "VSSL", "VTL", "VARROC", "VBL", "VASA", "VASCONEQ", "VASWANI", "VCL", "VEDL", "VENKEYS", "VENUSREM", "VERA", "VERTOZ", "VESUVIUS", "VETO", "VGUARD", "VICEROY", "VIDEOIND", "VIDHIING", "VIJIFIN", "VIKASECO", "VIKASMCORP", "VIMTALABS", "VINATIORGA", "VINDHYATEL", "VINNY", "VINYLINDIA", "VIPCLOTHNG", "VIPIND", "VIPULLTD", "VISASTEEL", "VIVIDHA", "VISAKAIND", "VISHNU", "VISHWARAJ", "VIVIMEDLAB", "VLSFINANCE", "VMART", "IDEA", "VOLTAMP", "VOLTAS", "VRLLOG", "VSTIND", "WSI", "WABCOINDIA", "WALCHANNAG", "WANBURY", "WEALTH", "WEBELSOLAR", "WEIZMANIND", "WELCORP", "WELENT", "WELSPUNIND", "WELINV", "WENDT", "WSTCSTPAPR", "WHEELS", "WHIRLPOOL", "WILLAMAGOR", "WINDMACHIN", "WIPRO", "WOCKPHARMA", "WFL", "WONDERLA", "WORTH", "XCHANGING", "XELPMOC", "XPROINDIA", "YESBANK", "ZEEL", "ZEELEARN", "ZEEMEDIA", "ZENTEC", "ZENITHBIR", "ZENITHEXPO", "ZENSARTECH", "ZICOM", "ZODIACLOTH", "ZODIAC", "ZODJRDMKJ", "ZOTA", "ZUARI", "ZUARIGLOB", "ZYDUSWELL", "ZAGGLE"    

        ]

for ticker in tickers:
    try:
        predictor = StockEngine(ticker, 1000)
        resp = predictor.predict_today()
        print(ticker+" : "+resp["decision"])
    except :
        pass

