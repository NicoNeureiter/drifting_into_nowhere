#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals


# Paths to separate template files
CDRW_XML_TEMPLATE_PATH = 'data/templates/beast_1_rrw_fixeddrift_template.xml'
RDRW_XML_TEMPLATE_PATH = 'data/templates/beast_1_rrw_drift_template.xml'
RRW_XML_TEMPLATE_PATH = 'data/templates/beast_1_rrw_template.xml'
BROWNIAN_XML_TEMPLATE_PATH = 'data/templates/beast_1_brownian_template.xml'
LOCATION_TEMPLATE_PATH = 'data/templates/location_template.xml'


# Short templates defined as python strings
LOCATION_TEMPLATE = '''\
        <taxon id="{id}">
			<date value="{age}" direction="forwards" units="years"/>
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
# LOCATION_TEMPLATE = ' '*24 + '{id}={x} {y}'
# FEATURES_TEMPLATE = '    <sequence id="seq_{id}" taxon="{id}" totalcount="2" value="0"/>\n'
# LEAF_TRAIT = '\t\t<leafTrait taxon="{id}"><parameter id="{id}.trait"/></leafTrait>\n'
# GEO_PRIOR_REF = '\t\t\t< flatGeoSpatialPrior idref = "{id}_region"/>\n'

# MONOPHYLY_STATISTIC = '\t<monophylyStatistic id="monophyly({id})"><mrca><taxa idref="{id}"/>'\
#     '</mrca><treeModel idref="treeModel"/>\t</monophylyStatistic>\n'
# GEO_PRIOR = '''\
#   <flatGeoSpatialPrior id="{id}_region" taxon="{id}" kmlFileName="{kml_path}" inside="true" union="true">
#       <data>
#           <parameter idref="{id}.location"/>
#       </data>
#   </flatGeoSpatialPrior>
# '''
SPHERICAL = ' greatCircleDistance="true"'

HEIGHT_OPERATORS = '''
        <scaleOperator scaleFactor="0.75" weight="3">
            <parameter idref="treeModel.rootHeight"/>
        </scaleOperator>
        <uniformOperator weight="30">
            <parameter idref="treeModel.internalNodeHeights"/>
        </uniformOperator>
'''
TREE_OPERATORS = '''
        <subtreeSlide size="1.0" gaussian="true" weight="15">
            <treeModel idref="treeModel"/>
        </subtreeSlide>
        <narrowExchange weight="15">
            <treeModel idref="treeModel"/>
        </narrowExchange>
        <wideExchange weight="3">
            <treeModel idref="treeModel"/>
        </wideExchange>
        <wilsonBalding weight="3">
            <treeModel idref="treeModel"/>
        </wilsonBalding>
'''


################################################################################
# Nexus & locations csv templates
################################################################################

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


################################################################################
# Not used for now
################################################################################

FOSSILS = '\t<taxa id="{id}">\n{descendants}\n\t</taxa>\n'
FOSSIL_CHILD = '\t\t<taxon idref="{id}"/>\n'

