---
layout: home
permalink: /
image:
  feature: feature_image/DJ.jpg
---
>Portfolio

<div class="tiles">
{% for post in site.categories.project %}
  {% include post-grid-noexcerpt.html %}
{% endfor %}

{% for post in site.categories.music %}
  {% include post-grid-noexcerpt.html %}
{% endfor %}
</div><!-- /.tiles -->




