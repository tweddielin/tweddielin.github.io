---
layout: sub_music
title: "Music"
date: 2014-06-02T09:44:20-04:00
modified:
excerpt:
image:
  feature: feature_image/synthesizer.jpg
  teaser:
  thumb:
share: false
ads: false
---

<div class="tiles">
{% for post in site.categories.music %}
  {% include post-grid.html %}
{% endfor %}
</div><!-- /.tiles -->
