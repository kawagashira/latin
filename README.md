## DOWNLOAD TEXT FILE ##
from Perseus Online
Metamorphoses Book-I by Ovid as in XML file
http://www.perseus.tufts.edu/hopper/xmlchunk?doc=Perseus%3Atext%3A1999.02.0029%3Abook%3D1

~/data/metamorphoses/book1.xml                                              (1)

## XML FILE FORMAT AS FOLLOWS ##
```
<?xml version="1.0" encoding="utf-8"?>
<TEI.2><text><body><div1 type="Book" n="1" org="uniform" sample="complete">
<milestone n="1" unit="card" />
<milestone ed="Magnus" n="Invocatio." unit="tale" />
<milestone ed="More" n="Invocation" unit="tale" />

<milestone ed="P" unit="para" /><l>In nova fert animus mutatas dicere formas</l>
<l>corpora; di, coeptis (nam vos mutastis et illas)</l>
<l>adspirate meis primaque ab origine mundi</l>
<l>ad mea perpetuum deducite tempora carmen.</l>

<milestone n="5" unit="card" />
<milestone ed="Magnus" n="Mundi origo." unit="tale" />

<milestone ed="P" unit="para" /><l n="5">Ante mare et terras et quod tegit omnia caelum</l>
<l>unus erat toto naturae vultus in orbe,</l>
<l>quem dixere chaos: rudis indigestaque moles</l>
<l>nec quicquam nisi pondus iners congestaque eodem</l>
<l>non bene iunctarum discordia semina rerum.</l>
<l n="10">nullus adhuc mundo praebebat lumina Titan,</l>
<l>nec nova crescendo reparabat cornua Phoebe,</l>
```

## XML PARSER ##
import xml.etree.ElementTree as ET

## PARSE XML TO TEXT ##
parse_xml() in parse_xml.py
