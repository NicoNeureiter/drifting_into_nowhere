#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals


MONOPHYLY_STATISTIC = '\t<monophylyStatistic id="monophyly({id})"><mrca><taxa idref="{id}"/>'\
    '</mrca><treeModel idref="treeModel"/>\t</monophylyStatistic>\n'


GEO_PRIOR = '''\
	<flatGeoSpatialPrior id="{id}_region" taxon="{id}" kmlFileName="{kml_path}" inside="true" union="true">
		<data>
			<parameter idref="{id}.location"/>
		</data>
	</flatGeoSpatialPrior>\
'''

LEAF_TRAIT = '\t\t<leafTrait taxon="{id}"><parameter id="{id}.trait"/></leafTrait>\n'

GEO_PRIOR_REF = '\t\t\t< flatGeoSpatialPrior idref = "{id}_region"/>\n'