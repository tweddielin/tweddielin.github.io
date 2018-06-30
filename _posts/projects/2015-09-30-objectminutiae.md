---
layout: article
title: "ObjectMinutiae"
date: 2015-09-30T23:14:02-04:00
modified:
categories: project
excerpt: A framework for authenticating different objects or materials via extracting and matching their fingerprints.
tags: [video]
ads: false
image:
  feature:
  teaser: project_image/2015-09-30-objectminutiae/device400x400.jpg
---

##### A framework for authenticating different objects or materials via extracting and matching their fingerprints.


This is a work which is collaborated with <a href="https://bitmark.com/"><font color="#3399FF">Bitmark Inc.</font></a> during the time I was working as a research assistant in Acamemia Sinica. Bitmark is a company focused on building a decentralized property system of the future. It is a global property system that can adequately describe ownership rights by giving all assets a digital identity that is inseparable from ownership. Digital assets can be uniquely identified using cryptographically hash functions. Physical unclonable functions (PUFs) provide a method to uniquely identify physical assets. This work is about how to identify physical assets such as works of art, official documents or luxury goods for their notorious forgery in the market. Unlike digital assets, it's much harder to identify physical assets which often require certain procedures (e.g., watermark, bar code, RFID, etc.) or even expert verifcation of an item's authenticity. Even though, it is still posiible to replicate or even tamper with such authentication processes. A desirable solution would be something akin to biometric methods for human identification and verifcation. Such a process would dientify distinctive patterns or key features that could be used to uniquely authenticate an item noninvasively. Taking advantage of object's textrual randomness in micro-scale, such distinctive patterns are able to be extrated, together with proper hashing or encryption techniques, the resulting features can be compact yet non-replicable, thereby securing the authentication process without requiring additional human verifcation.

<iframe src="https://www.youtube.com/embed/JgyGChTNy3E" frameborder="0"> </iframe>

<em>ObjectMinutiae</em> is a framework for authenticating different objects or materials via extracting and matching their fingerprints. Unlike common biometrics fingerprinting processes, which use patterns such as ridge ending and bifurcation points as the interest points, our work applies stereo photometric techniques for reconstructing objects’ local image regions that contain the surface texture information. The interest points of the recovered image regions can be detected and described by FAST and FREAK.

<figure class="half">
	<img src="/images/project_image/2015-09-30-objectminutiae/device900x600.jpg">
	<img src="/images/project_image/2015-09-30-objectminutiae/device_back900x600.jpg">
	<figcaption style="text-align:center">The prototype of the device.</figcaption>
</figure>

Together with dimension reduction and hashing techniques, our proposed system is able to perform object verification using compact image features. With neutral and different torturing conditions, preliminary results on multiple types of papers support the use of our framework for practical object authentication tasks.

<figure>
	<img src="/images/project_image/2015-09-30-objectminutiae/pipeline.png" style="width:75%;height:75%;">
	<figcaption style="text-align:center">Proposed framework.</figcaption>
</figure>

We currently consider an Apple iPhone5 with Olloclip 21X macro lens and a 3D-printed photometric stereo mask for implementation. For our device, we set up 4 white light LEDs mounted inside the mask (at 0, 90, 180, and 270 degrees) in such a way that each LED can be controlled independently. The output gradient image via stereo photometric techniques would contain detailed information about the surface texture. Once the gradient image of the ROI is derived, the local interest points (i.e., keypoints) are identified and described by FAST and FREAK descriptors, respectively. We further apply random projection and locality- sensitive hashing, which allow us to encode the extracted descriptors and reduce their feature dimensions. In our work, we only keep fewer than 300 keypoints for each ROI, while each descriptor requires only 64 bits. Adding the locations of the keypoints results in a fingerprint size for the ROI that is only about 25K bits.

The paper can be <a href="http://dl.acm.org/citation.cfm?id=2807989"><font color="grey">downloaded</font></a> here.

<figure align="middle">
	<img src="/images/project_image/2015-09-30-objectminutiae/final60samples.png" style="width:50%;height:50%;">
	<figcaption style="text-align:center">Matching score distributions.</figcaption>
</figure>

<figure class="third">
	<img src="/images/project_image/2015-09-30-objectminutiae/acmmm1.jpg">
	<img src="/images/project_image/2015-09-30-objectminutiae/acmmm2.jpg">
	<img src="/images/project_image/2015-09-30-objectminutiae/acmmm3.jpg">
</figure>

**T.-Y. Lin**, C.-Y. F. Wang, S. Moss-Pultz, “ObjectMinutiae: Fingerprinting for Object Authentication”, ACM International Conference on Multimedia (MM’ 15), Brisbane, Australia, October 2015
