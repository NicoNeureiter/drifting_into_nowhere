#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals


# Paths to separate template files

XML_TEMPLATE_PATH = 'data/templates/beast_xml_template.xml'
LOCATION_TEMPLATE_PATH = 'data/templates/location_template.xml'


# Short templates defined as python strings
LEAF_TRAIT = '\t\t<leafTrait taxon="{id}"><parameter id="{id}.trait"/></leafTrait>\n'
GEO_PRIOR_REF = '\t\t\t< flatGeoSpatialPrior idref = "{id}_region"/>\n'

LOCATION_TEMPLATE = '''\
		<taxon id="{id}">
			<attr name="location">
				{x} {y}
			</attr>
		</taxon>
'''
FEATURES_TEMPLATE = '''\
		<sequence>
			<taxon idref="{id}"/>
			{features}
		</sequence>
'''

MONOPHYLY_STATISTIC = '\t<monophylyStatistic id="monophyly({id})"><mrca><taxa idref="{id}"/>'\
    '</mrca><treeModel idref="treeModel"/>\t</monophylyStatistic>\n'
GEO_PRIOR = '''\
	<flatGeoSpatialPrior id="{id}_region" taxon="{id}" kmlFileName="{kml_path}" inside="true" union="true">
		<data>
			<parameter idref="{id}.location"/>
		</data>
	</flatGeoSpatialPrior>\
'''

# Nexus & locations csv templates
LOCATIONS_CSV_TEMPLATE = 'traits\tx\ty\n{data}'
NEXUS_TEMPLATE = '''#NEXUS
BEGIN DATA;
    DIMENSIONS NTAX={n_societies} NCHAR={n_features};
    FORMAT DATATYPE=standard SYMBOLS={symbols} MISSING=? GAP=-;
    MATRIX
{data}\t;
END;

BEGIN TREES;
      TREE original_tree = {tree};
END;
'''
