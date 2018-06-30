---
layout: article
title: "GeoMonster"
date: 2015-05-31T23:14:02-04:00
modified:
categories: project
excerpt: An interactive visualization system to analyze and predict urban construction dynamics.
tags: [data]
ads: false
image:
  feature:
  teaser: project_image/2015-05-31-geomonster/geo400x400.png
---
##### An interactive visualization system to analyze and predict urban construction dynamics
<a href="http://junipertcy.info.s3.amazonaws.com/urbcomp/index.html"><font color="blue">Demo Website</font></a>

GeoMonster is an analytic and visualization system for citizens and government agencies to understand, track, and predict the construction dynamics in an urban area. It was started as a hackathon project of 2015 Taipei Open Data Hackathon and surprisingly ended as a workshop paper in ACM SIGKDD Workshop on Urban Computing 2015. In the hackathon, Tzu-Chi, Ching-Yuan, and I were brainstorming about what is the urgent or serious issue that need to be solved by linking urban data and its geometric information with citizens in the city. <!--more-->Also, there are so many different types of urban data now opened by Taipei City Government (<a href="http://data.taipei"><font color="blue">http://data.taipei</font></a>). It turned out that construction issue is really bothering. Recalling from our experiences and surveys, we found some interesting phenomena.


<figure>
	<img src="/images/project_image/2015-05-31-geomonster/geomonster.png">
	<figcaption>Snapshot of the system we made in the hackathon.</figcaption>
</figure>

That says, the city construction request records can collectively depict urban contruction events through not only their geographical and temporal dynamics of contruction requests but also the corresponding labeled causes of construction events. These data can be analyzed by dynmically grouping regions that have the similar patterns of contruction requests. For example, we might be able to identify some regions that both the powerline and water pipeline related constructions play the dominant cause of construction requests. The analysis might also help to unveil the underlying problems of infrastructures or services. For example, the relatively high frquency of reporting in water-pipe burst in a certain region might hint aging or strctural weakness of the water pipeline in that area. The prolonged durations of contruction might indicate inefficient constructions. In addition to the visual presentation and analytics, the urban contruction request data also offer some potentials to be used to build a predictive model that estimates the future construction requests of certain types in any urban area of interest.

<figure class="half">
	<img src="/images/project_image/2015-05-31-geomonster/geoframework.png">
	<img src="/images/project_image/2015-05-31-geomonster/geodata.png">
	<figcaption style="text-align:center">System framework and data table.</figcaption>
</figure>

<figure align="middle">
	<img src="/images/project_image/2015-05-31-geomonster/geocluster.png" style="width:80%;height:80%;">
	<figcaption style="text-align:center">Clustering-based Region Exploration.</figcaption>
</figure>

<figure align="middle">
	<img src="/images/project_image/2015-05-31-geomonster/geopredfig.png" style="width:50%;height:50%;">
	<figcaption style="text-align:center">Regional Predictive Analysis Result.</figcaption>
</figure>

T.-C. Yen, **T.-Y. Lin**, C.-Y. Yeh, H.-P. Hsieh, and C.-T. Li, “An Interactive Visualization System to Analyze and Predict Urban Construction Dynamics”, ACM SIGKDD International Workshop on Urban Computing (UrbComp’ 15), in conjunction with KDD 2015, Sydney, Australia, Aug. 10, 2015.
