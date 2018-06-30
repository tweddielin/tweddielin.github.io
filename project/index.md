---
layout: sub_projects
title: "Project"
date: 2014-06-02T09:44:20-04:00
modified:
excerpt:
image:
  feature: feature_image/electronics.jpg
  teaser:
  thumb:
share: false
ads: false
---

<div class="tiles">
{% for post in site.categories.project %}
  {% include post-grid-nocategories.html %}
{% endfor %}
</div><!-- /.tiles -->
