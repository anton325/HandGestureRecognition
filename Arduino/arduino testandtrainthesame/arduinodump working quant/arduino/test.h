#ifndef  TEST_H
#define  TEST_H
#include  <avr/pgmspace.h>
const  int  numberOfTests  =  32;
const  int  testset[]  PROGMEM  =  {
  677 ,684 ,669 ,714 ,667 ,700 ,725 ,680 ,661 ,
666 ,667 ,623 ,711 ,657 ,680 ,726 ,680 ,656 ,
648 ,639 ,562 ,700 ,632 ,633 ,719 ,665 ,630 ,
617 ,598 ,486 ,683 ,599 ,582 ,716 ,654 ,603 ,
579 ,551 ,399 ,655 ,551 ,513 ,701 ,622 ,552 ,
513 ,478 ,312 ,613 ,491 ,422 ,682 ,586 ,501 ,
433 ,391 ,251 ,555 ,410 ,333 ,648 ,530 ,422 ,
333 ,299 ,211 ,464 ,312 ,254 ,602 ,450 ,330 ,
270 ,250 ,184 ,372 ,233 ,213 ,522 ,352 ,242 ,
237 ,235 ,192 ,299 ,191 ,177 ,419 ,255 ,180 ,
228 ,240 ,267 ,277 ,201 ,196 ,344 ,216 ,157 ,
262 ,289 ,386 ,269 ,219 ,256 ,290 ,208 ,167 ,
338 ,383 ,507 ,302 ,268 ,366 ,259 ,212 ,197 ,
424 ,472 ,596 ,375 ,353 ,478 ,272 ,234 ,239 ,
501 ,543 ,645 ,470 ,454 ,579 ,339 ,321 ,342 ,
572 ,602 ,682 ,560 ,536 ,639 ,436 ,418 ,445 ,
614 ,638 ,698 ,613 ,581 ,670 ,525 ,507 ,523 ,
650 ,670 ,713 ,657 ,622 ,696 ,591 ,564 ,570 ,
666 ,683 ,716 ,683 ,643 ,705 ,636 ,603 ,604 ,
678 ,695 ,721 ,703 ,665 ,717 ,680 ,643 ,635 ,
693 ,710 ,730 ,726 ,687 ,730 ,734 ,694 ,679 ,
693 ,710 ,730 ,724 ,685 ,728 ,730 ,689 ,673 ,
693 ,707 ,671 ,725 ,682 ,652 ,732 ,686 ,577 ,
688 ,627 ,457 ,719 ,580 ,406 ,724 ,576 ,345 ,
596 ,416 ,206 ,630 ,359 ,163 ,631 ,359 ,134 ,
377 ,149 ,61 ,415 ,115 ,67 ,417 ,135 ,56 ,
115 ,51 ,50 ,139 ,39 ,51 ,149 ,51 ,33 ,
51 ,48 ,48 ,68 ,31 ,44 ,69 ,35 ,25 ,
42 ,49 ,45 ,53 ,28 ,41 ,48 ,28 ,22 ,
44 ,45 ,38 ,50 ,26 ,36 ,43 ,25 ,20 ,
43 ,42 ,36 ,47 ,24 ,34 ,40 ,23 ,18 ,
40 ,45 ,90 ,42 ,22 ,59 ,37 ,22 ,23 ,
46 ,67 ,266 ,39 ,29 ,165 ,35 ,25 ,55 ,
62 ,164 ,558 ,44 ,84 ,456 ,37 ,53 ,200 ,
141 ,398 ,662 ,105 ,286 ,613 ,77 ,182 ,428 ,
325 ,561 ,709 ,272 ,477 ,692 ,189 ,376 ,581 ,
496 ,649 ,725 ,460 ,600 ,720 ,355 ,546 ,653 ,
605 ,690 ,728 ,604 ,659 ,727 ,529 ,649 ,673 ,
662 ,708 ,730 ,683 ,684 ,730 ,661 ,687 ,676 ,
688 ,709 ,730 ,716 ,683 ,727 ,719 ,688 ,674 ,
688 ,709 ,729 ,716 ,685 ,728 ,711 ,691 ,676 ,
665 ,708 ,731 ,681 ,683 ,731 ,633 ,681 ,678 ,
591 ,690 ,729 ,581 ,651 ,727 ,470 ,620 ,670 ,
420 ,635 ,724 ,382 ,565 ,717 ,250 ,476 ,633 ,
225 ,487 ,698 ,189 ,365 ,667 ,107 ,249 ,513 ,
151 ,272 ,602 ,120 ,154 ,507 ,64 ,88 ,283 ,
97 ,159 ,362 ,76 ,69 ,234 ,47 ,38 ,87 ,
50 ,101 ,162 ,49 ,41 ,81 ,39 ,27 ,31 ,
40 ,51 ,87 ,45 ,26 ,46 ,38 ,23 ,21 ,
33 ,45 ,44 ,38 ,25 ,36 ,34 ,23 ,19 ,
28 ,36 ,38 ,34 ,21 ,34 ,33 ,21 ,18 ,
31 ,32 ,37 ,36 ,19 ,33 ,35 ,20 ,16 ,
82 ,34 ,31 ,71 ,20 ,30 ,62 ,21 ,16 ,
217 ,75 ,29 ,182 ,42 ,30 ,150 ,35 ,16 ,
401 ,202 ,65 ,390 ,122 ,50 ,354 ,88 ,26 ,
585 ,372 ,204 ,606 ,289 ,123 ,595 ,235 ,62 ,
672 ,564 ,404 ,701 ,513 ,293 ,704 ,479 ,168 ,
689 ,676 ,595 ,721 ,646 ,545 ,729 ,641 ,419 ,
693 ,707 ,700 ,723 ,680 ,688 ,730 ,682 ,605 ,
691 ,707 ,725 ,724 ,685 ,724 ,733 ,692 ,669 ,
684 ,700 ,724 ,708 ,669 ,719 ,689 ,653 ,645 ,
668 ,688 ,718 ,690 ,652 ,712 ,649 ,621 ,622 ,
650 ,674 ,714 ,661 ,625 ,700 ,591 ,570 ,582 ,
616 ,644 ,703 ,615 ,585 ,678 ,518 ,507 ,530 ,
561 ,599 ,682 ,550 ,529 ,639 ,424 ,416 ,450 ,
495 ,547 ,652 ,471 ,455 ,588 ,338 ,325 ,355 ,
418 ,475 ,601 ,383 ,360 ,495 ,300 ,261 ,270 ,
342 ,397 ,525 ,329 ,292 ,398 ,316 ,245 ,226 ,
286 ,319 ,422 ,312 ,250 ,300 ,355 ,242 ,199 ,
269 ,278 ,318 ,329 ,233 ,238 ,419 ,253 ,187 ,
287 ,277 ,242 ,357 ,224 ,211 ,494 ,306 ,211 ,
325 ,288 ,217 ,425 ,267 ,231 ,574 ,397 ,274 ,
384 ,328 ,224 ,502 ,341 ,270 ,624 ,476 ,351 ,
460 ,400 ,254 ,567 ,420 ,334 ,658 ,537 ,424 ,
524 ,474 ,304 ,616 ,491 ,409 ,686 ,585 ,492 ,
579 ,540 ,377 ,654 ,545 ,489 ,702 ,618 ,539 ,
618 ,589 ,457 ,681 ,591 ,559 ,715 ,647 ,589 ,
648 ,631 ,538 ,699 ,628 ,618 ,720 ,665 ,625 ,
667 ,663 ,608 ,711 ,655 ,669 ,725 ,678 ,651 ,
678 ,685 ,665 ,714 ,668 ,699 ,727 ,681 ,661 ,
666 ,666 ,618 ,712 ,658 ,676 ,726 ,680 ,655 ,
646 ,635 ,552 ,699 ,631 ,628 ,720 ,667 ,629 ,
615 ,595 ,473 ,681 ,593 ,572 ,713 ,649 ,595 ,
567 ,537 ,379 ,649 ,544 ,498 ,701 ,621 ,548 ,
498 ,457 ,295 ,601 ,474 ,401 ,675 ,573 ,482 ,
402 ,356 ,235 ,529 ,380 ,306 ,639 ,513 ,399 ,
295 ,267 ,193 ,429 ,273 ,236 ,577 ,415 ,298 ,
211 ,203 ,151 ,321 ,198 ,187 ,479 ,307 ,213 ,
176 ,180 ,140 ,232 ,151 ,145 ,353 ,212 ,153 ,
180 ,194 ,192 ,198 ,141 ,135 ,252 ,163 ,114 ,
221 ,243 ,317 ,214 ,171 ,188 ,199 ,148 ,108 ,
299 ,342 ,452 ,258 ,226 ,296 ,199 ,163 ,145 ,
392 ,442 ,561 ,333 ,314 ,426 ,230 ,198 ,197 ,
468 ,514 ,624 ,422 ,401 ,523 ,281 ,260 ,272 ,
530 ,568 ,659 ,501 ,482 ,596 ,359 ,343 ,361 ,
578 ,607 ,682 ,565 ,540 ,640 ,443 ,425 ,445 ,
618 ,641 ,701 ,614 ,581 ,669 ,517 ,497 ,509 ,
643 ,663 ,708 ,648 ,611 ,688 ,572 ,548 ,555 ,
662 ,681 ,715 ,676 ,636 ,701 ,620 ,589 ,590 ,
675 ,692 ,719 ,698 ,660 ,714 ,666 ,630 ,623 ,
696 ,712 ,731 ,727 ,689 ,730 ,734 ,693 ,674 ,
693 ,707 ,679 ,725 ,682 ,665 ,731 ,688 ,599 ,
693 ,679 ,535 ,724 ,647 ,476 ,731 ,649 ,388 ,
675 ,537 ,287 ,709 ,481 ,222 ,715 ,477 ,168 ,
559 ,284 ,103 ,597 ,220 ,96 ,605 ,229 ,81 ,
293 ,104 ,42 ,320 ,76 ,48 ,337 ,96 ,39 ,
105 ,42 ,36 ,119 ,32 ,40 ,136 ,45 ,26 ,
43 ,36 ,38 ,57 ,25 ,37 ,64 ,31 ,21 ,
33 ,37 ,44 ,43 ,23 ,37 ,44 ,25 ,20 ,
33 ,44 ,33 ,40 ,25 ,33 ,39 ,24 ,19 ,
40 ,38 ,29 ,42 ,22 ,31 ,38 ,23 ,18 ,
35 ,33 ,31 ,39 ,20 ,31 ,36 ,22 ,18 ,
30 ,34 ,87 ,35 ,20 ,48 ,35 ,21 ,19 ,
30 ,63 ,311 ,35 ,26 ,170 ,35 ,23 ,48 ,
51 ,179 ,566 ,42 ,89 ,439 ,36 ,51 ,177 ,
144 ,410 ,671 ,102 ,274 ,621 ,70 ,159 ,416 ,
333 ,580 ,718 ,257 ,491 ,708 ,167 ,363 ,619 ,
515 ,668 ,728 ,466 ,625 ,726 ,331 ,586 ,669 ,
622 ,703 ,729 ,626 ,676 ,728 ,567 ,676 ,674 ,
685 ,711 ,732 ,712 ,687 ,731 ,711 ,690 ,677 ,
695 ,712 ,732 ,725 ,689 ,731 ,725 ,693 ,678 ,
668 ,707 ,729 ,681 ,682 ,728 ,627 ,682 ,676 ,
599 ,692 ,729 ,579 ,653 ,728 ,448 ,615 ,672 ,
432 ,640 ,723 ,369 ,562 ,715 ,227 ,450 ,625 ,
218 ,510 ,703 ,165 ,367 ,670 ,92 ,232 ,500 ,
108 ,272 ,631 ,68 ,143 ,531 ,44 ,78 ,271 ,
78 ,115 ,426 ,45 ,40 ,249 ,36 ,29 ,79 ,
46 ,78 ,131 ,41 ,24 ,63 ,36 ,21 ,27 ,
52 ,53 ,59 ,51 ,23 ,37 ,43 ,22 ,20 ,
46 ,58 ,47 ,44 ,31 ,35 ,35 ,25 ,19 ,
31 ,53 ,69 ,34 ,27 ,47 ,31 ,22 ,21 ,
28 ,35 ,60 ,33 ,19 ,40 ,31 ,19 ,18 ,
89 ,31 ,34 ,77 ,19 ,31 ,58 ,19 ,16 ,
319 ,78 ,30 ,262 ,48 ,30 ,187 ,32 ,15 ,
532 ,312 ,90 ,537 ,208 ,70 ,504 ,126 ,30 ,
659 ,509 ,333 ,684 ,426 ,237 ,682 ,346 ,97 ,
687 ,637 ,540 ,718 ,594 ,433 ,724 ,569 ,243 ,
692 ,693 ,643 ,723 ,663 ,595 ,729 ,660 ,447 ,
691 ,705 ,698 ,723 ,681 ,684 ,731 ,687 ,595 ,
696 ,711 ,725 ,726 ,687 ,721 ,732 ,689 ,657,
694 ,710 ,730 ,725 ,686 ,729 ,732 ,692 ,678 ,
696 ,712 ,732 ,727 ,690 ,731 ,730 ,689 ,675 ,
695 ,712 ,732 ,721 ,682 ,726 ,625 ,594 ,590 ,
689 ,707 ,728 ,643 ,610 ,680 ,238 ,232 ,259 ,
607 ,643 ,690 ,336 ,313 ,428 ,36 ,24 ,26 ,
301 ,363 ,445 ,55 ,28 ,45 ,17 ,9 ,9 ,
39 ,47 ,53 ,25 ,10 ,15 ,12 ,6 ,6 ,
15 ,16 ,15 ,17 ,6 ,9 ,10 ,4 ,4 ,
10 ,10 ,8 ,14 ,4 ,7 ,9 ,4 ,4 ,
8 ,8 ,6 ,12 ,3 ,6 ,9 ,6 ,3 ,
7 ,6 ,5 ,11 ,7 ,6 ,9 ,7 ,2 ,
7 ,12 ,5 ,12 ,9 ,5 ,46 ,6 ,2 ,
7 ,17 ,4 ,17 ,6 ,4 ,335 ,54 ,3 ,
7 ,11 ,3 ,136 ,15 ,4 ,613 ,306 ,38 ,
50 ,11 ,3 ,466 ,139 ,20 ,710 ,568 ,290 ,
340 ,99 ,6 ,669 ,455 ,160 ,728 ,666 ,554 ,
600 ,428 ,65 ,718 ,639 ,523 ,733 ,690 ,659 ,
680 ,645 ,395 ,724 ,680 ,694 ,731 ,690 ,673 ,
689 ,699 ,655 ,724 ,684 ,725 ,733 ,693 ,678 ,
696 ,712 ,727 ,727 ,689 ,731 ,734 ,693 ,678 ,
697 ,714 ,733 ,727 ,690 ,732 ,733 ,693 ,678 ,
692 ,695 ,650 ,726 ,687 ,725 ,733 ,694 ,680 ,
626 ,507 ,300 ,722 ,655 ,618 ,733 ,692 ,670 ,
301 ,137 ,42 ,649 ,409 ,272 ,730 ,662 ,557 ,
45 ,20 ,10 ,315 ,93 ,44 ,672 ,437 ,234 ,
14 ,15 ,6 ,63 ,12 ,11 ,334 ,100 ,35 ,
10 ,32 ,5 ,22 ,7 ,7 ,46 ,11 ,8 ,
6 ,11 ,3 ,16 ,19 ,4 ,16 ,6 ,4 ,
5 ,6 ,2 ,11 ,9 ,3 ,11 ,13 ,3 ,
4 ,4 ,1 ,9 ,3 ,3 ,8 ,11 ,2 ,
3 ,3 ,1 ,7 ,1 ,2 ,6 ,3 ,1 ,
3 ,3 ,0 ,6 ,1 ,1 ,5 ,2 ,1 ,
3 ,2 ,0 ,6 ,0 ,1 ,5 ,1 ,0 ,
39 ,40 ,42 ,5 ,1 ,1 ,3 ,1 ,0 ,
479 ,491 ,581 ,35 ,25 ,42 ,3 ,1 ,0 ,
678 ,691 ,718 ,338 ,297 ,373 ,10 ,3 ,2 ,
692 ,708 ,728 ,690 ,643 ,698 ,299 ,226 ,215 ,
694 ,710 ,730 ,723 ,683 ,727 ,667 ,592 ,574 ,
693 ,710 ,731 ,725 ,686 ,730 ,731 ,687 ,672 ,
695 ,712 ,732 ,725 ,686 ,729 ,730 ,689 ,674 ,
696 ,713 ,733 ,726 ,688 ,730 ,732 ,692 ,678 ,
694 ,711 ,724 ,726 ,688 ,725 ,733 ,694 ,661 ,
694 ,711 ,332 ,726 ,688 ,241 ,734 ,694 ,196 ,
695 ,473 ,442 ,727 ,408 ,28 ,733 ,387 ,14 ,
531 ,367 ,78 ,443 ,48 ,15 ,430 ,17 ,5 ,
502 ,93 ,11 ,124 ,20 ,6 ,38 ,4 ,3 ,
92 ,15 ,7 ,59 ,5 ,11 ,12 ,2 ,2 ,
19 ,8 ,4 ,20 ,4 ,5 ,7 ,2 ,2 ,
10 ,7 ,3 ,13 ,3 ,4 ,7 ,2 ,1 ,
8 ,5 ,2 ,16 ,2 ,3 ,9 ,2 ,0 ,
7 ,4 ,2 ,11 ,2 ,3 ,8 ,2 ,0 ,
5 ,4 ,1 ,9 ,1 ,2 ,7 ,2 ,0 ,
6 ,4 ,2 ,9 ,1 ,2 ,8 ,1 ,0 ,
5 ,4 ,89 ,9 ,1 ,4 ,8 ,1 ,0 ,
7 ,8 ,695 ,9 ,1 ,462 ,7 ,1 ,96 ,
6 ,335 ,728 ,11 ,69 ,722 ,10 ,2 ,607 ,
11 ,696 ,732 ,11 ,594 ,730 ,10 ,345 ,673 ,
585 ,708 ,730 ,249 ,681 ,728 ,124 ,675 ,677 ,
689 ,710 ,731 ,706 ,686 ,730 ,612 ,690 ,678 ,
695 ,712 ,733 ,725 ,687 ,731 ,729 ,690 ,677 ,
697 ,713 ,733 ,727 ,689 ,731 ,733 ,693 ,678 ,
694 ,710 ,730 ,726 ,688 ,730 ,729 ,694 ,680 ,
692 ,712 ,731 ,666 ,688 ,730 ,409 ,692 ,678 ,
228 ,712 ,732 ,178 ,680 ,732 ,50 ,511 ,680 ,
24 ,521 ,730 ,32 ,70 ,721 ,18 ,23 ,633 ,
12 ,30 ,537 ,19 ,12 ,415 ,11 ,8 ,77 ,
8 ,11 ,66 ,14 ,5 ,25 ,10 ,4 ,11 ,
7 ,7 ,11 ,12 ,4 ,10 ,9 ,4 ,5 ,
6 ,5 ,5 ,10 ,2 ,5 ,8 ,3 ,3 ,
5 ,5 ,4 ,9 ,2 ,4 ,8 ,3 ,3 ,
5 ,5 ,3 ,8 ,1 ,4 ,7 ,2 ,2 ,
5 ,4 ,3 ,7 ,1 ,4 ,5 ,2 ,1 ,
38 ,4 ,3 ,8 ,1 ,3 ,5 ,1 ,1 ,
187 ,21 ,2 ,14 ,1 ,2 ,7 ,1 ,0 ,
454 ,183 ,2 ,426 ,5 ,2 ,436 ,2 ,0 ,
685 ,210 ,189 ,715 ,8 ,6 ,722 ,7 ,0 ,
692 ,660 ,359 ,722 ,628 ,19 ,731 ,620 ,3 ,
694 ,708 ,520 ,725 ,683 ,364 ,732 ,687 ,301 ,
695 ,710 ,725 ,726 ,686 ,721 ,732 ,690 ,658 ,
693 ,709 ,729 ,726 ,688 ,729 ,735 ,694 ,678 ,
695 ,712 ,731 ,725 ,687 ,729 ,732 ,692 ,677 ,
694 ,711 ,731 ,727 ,689 ,731 ,735 ,695 ,681 ,
696 ,712 ,732 ,726 ,688 ,730 ,710 ,670 ,660 ,
693 ,709 ,729 ,704 ,662 ,718 ,434 ,403 ,450 ,
664 ,684 ,720 ,452 ,413 ,563 ,116 ,99 ,117 ,
339 ,392 ,524 ,140 ,110 ,169 ,26 ,16 ,19 ,
91 ,113 ,167 ,35 ,15 ,25 ,15 ,8 ,9 ,
20 ,21 ,21 ,21 ,8 ,13 ,12 ,5 ,6 ,
13 ,13 ,11 ,16 ,5 ,9 ,10 ,4 ,4 ,
9 ,9 ,8 ,14 ,4 ,8 ,12 ,4 ,3 ,
8 ,8 ,7 ,14 ,5 ,7 ,14 ,5 ,3 ,
8 ,11 ,6 ,15 ,6 ,6 ,106 ,11 ,2 ,
10 ,16 ,6 ,57 ,9 ,6 ,381 ,114 ,13 ,
34 ,17 ,5 ,258 ,61 ,15 ,660 ,373 ,114 ,
184 ,60 ,9 ,577 ,257 ,82 ,719 ,611 ,365 ,
492 ,244 ,43 ,699 ,547 ,285 ,731 ,678 ,590 ,
644 ,529 ,169 ,719 ,653 ,570 ,731 ,688 ,658 ,
683 ,656 ,443 ,725 ,680 ,691 ,734 ,693 ,676 ,
694 ,701 ,643 ,725 ,686 ,723 ,732 ,692 ,676 ,
692 ,706 ,709 ,725 ,686 ,728 ,733 ,693 ,679 ,
697 ,713 ,729 ,727 ,689 ,731 ,733 ,693 ,679 ,
692 ,699 ,648 ,725 ,687 ,725 ,734 ,694 ,680 ,
677 ,629 ,407 ,726 ,679 ,673 ,733 ,693 ,676 ,
551 ,397 ,131 ,710 ,602 ,463 ,732 ,685 ,632 ,
256 ,112 ,30 ,610 ,355 ,164 ,723 ,632 ,457 ,
49 ,22 ,10 ,324 ,92 ,40 ,658 ,413 ,189 ,
16 ,18 ,7 ,84 ,16 ,12 ,381 ,130 ,42 ,
11 ,16 ,6 ,27 ,13 ,8 ,92 ,21 ,10 ,
7 ,9 ,5 ,19 ,10 ,7 ,21 ,12 ,5 ,
7 ,6 ,4 ,14 ,5 ,6 ,16 ,12 ,4 ,
5 ,5 ,3 ,10 ,2 ,5 ,10 ,5 ,3 ,
4 ,4 ,2 ,8 ,2 ,4 ,8 ,3 ,2 ,
4 ,4 ,2 ,7 ,1 ,3 ,6 ,2 ,2 ,
10 ,13 ,14 ,6 ,1 ,2 ,6 ,2 ,1 ,
231 ,250 ,302 ,9 ,3 ,6 ,5 ,1 ,1 ,
588 ,610 ,668 ,247 ,227 ,306 ,6 ,2 ,1 ,
686 ,702 ,725 ,613 ,575 ,651 ,233 ,175 ,179 ,
693 ,710 ,731 ,718 ,679 ,723 ,621 ,560 ,546 ,
692 ,709 ,729 ,723 ,684 ,727 ,725 ,682 ,664 ,
696 ,712 ,733 ,727 ,689 ,732 ,733 ,692 ,677 ,
697 ,713 ,734 ,728 ,690 ,732 ,734 ,694 ,680 ,
693 ,710 ,730 ,726 ,687 ,730 ,734 ,694 ,680 ,
696 ,713 ,607 ,727 ,690 ,604 ,733 ,693 ,552 ,
693 ,672 ,101 ,725 ,650 ,98 ,733 ,663 ,63 ,
695 ,361 ,19 ,727 ,344 ,20 ,733 ,360 ,15 ,
392 ,31 ,8 ,448 ,20 ,9 ,459 ,21 ,6 ,
61 ,12 ,5 ,100 ,7 ,5 ,98 ,7 ,2 ,
16 ,7 ,4 ,26 ,3 ,4 ,18 ,3 ,2 ,
8 ,8 ,3 ,15 ,3 ,4 ,8 ,2 ,2 ,
7 ,6 ,3 ,12 ,2 ,3 ,8 ,2 ,1 ,
7 ,5 ,2 ,11 ,2 ,3 ,7 ,2 ,1 ,
6 ,5 ,2 ,11 ,2 ,2 ,7 ,2 ,1 ,
6 ,4 ,2 ,10 ,1 ,2 ,7 ,2 ,0 ,
6 ,4 ,63 ,9 ,1 ,4 ,8 ,2 ,0 ,
6 ,6 ,429 ,10 ,1 ,319 ,8 ,1 ,94 ,
6 ,388 ,725 ,12 ,206 ,700 ,10 ,10 ,459 ,
352 ,693 ,730 ,48 ,576 ,728 ,11 ,428 ,667 ,
650 ,709 ,731 ,558 ,681 ,728 ,382 ,669 ,674 ,
689 ,709 ,730 ,718 ,686 ,729 ,697 ,691 ,679 ,
696 ,713 ,733 ,726 ,689 ,731 ,730 ,691 ,678 ,
696 ,713 ,733 ,728 ,691 ,733 ,735 ,695 ,681 ,
695 ,711 ,731 ,725 ,686 ,729 ,725 ,692 ,678 ,
687 ,712 ,732 ,671 ,690 ,732 ,548 ,688 ,681 ,
531 ,706 ,731 ,185 ,627 ,730 ,42 ,535 ,675 ,
40 ,552 ,730 ,41 ,250 ,705 ,22 ,34 ,548 ,
19 ,60 ,595 ,24 ,17 ,457 ,14 ,11 ,76 ,
13 ,18 ,157 ,18 ,8 ,36 ,12 ,7 ,15 ,
10 ,12 ,19 ,15 ,5 ,14 ,11 ,5 ,7 ,
8 ,9 ,10 ,13 ,4 ,9 ,10 ,5 ,5 ,
7 ,8 ,8 ,11 ,3 ,7 ,9 ,4 ,4 ,
7 ,7 ,6 ,9 ,3 ,6 ,7 ,3 ,3 ,
8 ,7 ,6 ,9 ,2 ,6 ,7 ,2 ,3 ,
208 ,7 ,5 ,43 ,2 ,5 ,8 ,2 ,2 ,
348 ,122 ,5 ,230 ,14 ,5 ,317 ,3 ,2 ,
600 ,239 ,21 ,609 ,18 ,5 ,655 ,8 ,2 ,
689 ,548 ,273 ,720 ,493 ,14 ,729 ,512 ,3 ,
695 ,695 ,365 ,726 ,673 ,206 ,731 ,680 ,136 ,
692 ,708 ,684 ,725 ,684 ,670 ,732 ,691 ,612 ,
696 ,711 ,730 ,726 ,687 ,727 ,732 ,691 ,672 ,
693 ,709 ,729 ,726 ,688 ,730 ,735 ,695 ,679 ,
693 ,709 ,730 ,725 ,687 ,729 ,733 ,694 ,680 ,
697 ,714 ,734 ,728 ,689 ,732 ,728 ,684 ,669 ,
693 ,710 ,729 ,719 ,678 ,724 ,599 ,557 ,563 ,
690 ,704 ,728 ,623 ,577 ,667 ,170 ,145 ,172 ,
582 ,611 ,681 ,226 ,183 ,307 ,29 ,18 ,21 ,
250 ,305 ,466 ,47 ,23 ,37 ,17 ,9 ,11 ,
39 ,45 ,55 ,25 ,10 ,17 ,12 ,6 ,7 ,
18 ,18 ,18 ,18 ,7 ,11 ,10 ,5 ,5 ,
13 ,12 ,11 ,14 ,5 ,9 ,8 ,5 ,4 ,
9 ,10 ,9 ,12 ,5 ,8 ,9 ,4 ,4 ,
8 ,8 ,7 ,12 ,4 ,7 ,16 ,3 ,3 ,
8 ,8 ,7 ,18 ,3 ,6 ,15 ,3 ,2 ,
13 ,7 ,6 ,17 ,2 ,6 ,84 ,5 ,2 ,
13 ,6 ,5 ,56 ,4 ,6 ,562 ,137 ,12 ,
44 ,9 ,5 ,449 ,88 ,18 ,705 ,555 ,197 ,
383 ,116 ,12 ,681 ,489 ,156 ,730 ,674 ,564 ,
640 ,510 ,111 ,718 ,655 ,571 ,730 ,687 ,659 ,
686 ,675 ,495 ,726 ,686 ,709 ,734 ,694 ,678 ,
692 ,704 ,683 ,724 ,684 ,725 ,732 ,691 ,676 ,
696 ,711 ,726 ,727 ,690 ,731 ,734 ,694 ,679 ,
697 ,713 ,734 ,728 ,690 ,732 ,733 ,693 ,679 ,
693 ,710 ,728 ,726 ,688 ,730 ,734 ,694 ,680 ,
695 ,704 ,656 ,727 ,689 ,728 ,733 ,693 ,679 ,
650 ,579 ,296 ,723 ,668 ,632 ,733 ,694 ,673 ,
387 ,237 ,81 ,674 ,505 ,294 ,730 ,672 ,570 ,
114 ,58 ,21 ,403 ,185 ,83 ,683 ,518 ,273 ,
27 ,22 ,10 ,135 ,37 ,21 ,422 ,205 ,73 ,
16 ,15 ,7 ,40 ,13 ,11 ,137 ,40 ,15 ,
11 ,10 ,6 ,25 ,9 ,8 ,28 ,11 ,8 ,
7 ,7 ,5 ,17 ,5 ,6 ,18 ,7 ,5 ,
6 ,5 ,4 ,13 ,3 ,5 ,12 ,5 ,4 ,
5 ,5 ,3 ,10 ,2 ,4 ,9 ,3 ,3 ,
5 ,5 ,3 ,9 ,2 ,4 ,7 ,2 ,2 ,
89 ,86 ,129 ,8 ,1 ,4 ,6 ,1 ,1 ,
375 ,371 ,485 ,97 ,74 ,115 ,6 ,1 ,1 ,
672 ,682 ,713 ,422 ,360 ,485 ,91 ,60 ,61 ,
692 ,708 ,728 ,707 ,661 ,711 ,452 ,340 ,341 ,
693 ,709 ,730 ,724 ,685 ,729 ,717 ,666 ,648 ,
695 ,711 ,731 ,724 ,686 ,729 ,729 ,687 ,671 ,
693 ,709 ,730 ,726 ,688 ,730 ,734 ,693 ,679 ,
696 ,712 ,732 ,725 ,687 ,729 ,732 ,692 ,677 ,
694 ,710 ,725 ,727 ,688 ,727 ,734 ,694 ,675 ,
694 ,711 ,321 ,726 ,688 ,334 ,733 ,693 ,296 ,
697 ,631 ,30 ,727 ,634 ,31 ,733 ,657 ,23 ,
657 ,54 ,27 ,707 ,40 ,12 ,722 ,52 ,8 ,
263 ,37 ,9 ,326 ,12 ,7 ,334 ,12 ,4 ,
53 ,15 ,5 ,41 ,4 ,5 ,31 ,4 ,2 ,
23 ,8 ,4 ,18 ,3 ,4 ,12 ,3 ,2 ,
10 ,7 ,3 ,12 ,3 ,4 ,7 ,2 ,2 ,
8 ,6 ,3 ,11 ,2 ,3 ,7 ,2 ,1 ,
6 ,5 ,2 ,9 ,2 ,3 ,7 ,2 ,1 ,
6 ,4 ,2 ,9 ,1 ,2 ,8 ,2 ,0 ,
5 ,4 ,1 ,9 ,1 ,2 ,8 ,2 ,0 ,
5 ,3 ,59 ,7 ,0 ,7 ,7 ,1 ,0 ,
6 ,6 ,619 ,8 ,0 ,331 ,7 ,1 ,162 ,
6 ,500 ,728 ,10 ,208 ,722 ,8 ,3 ,591 ,
119 ,705 ,729 ,33 ,670 ,727 ,5 ,509 ,672 ,
680 ,709 ,731 ,559 ,685 ,730 ,239 ,684 ,677 ,
692 ,711 ,732 ,721 ,687 ,730 ,694 ,691 ,678 ,
692 ,709 ,730 ,723 ,685 ,728 ,730 ,691 ,678 ,
696 ,712 ,732 ,726 ,687 ,730 ,732 ,692 ,677 ,
693 ,709 ,730 ,726 ,687 ,730 ,729 ,694 ,680 ,
695 ,712 ,732 ,706 ,688 ,730 ,580 ,692 ,678 ,
603 ,711 ,731 ,458 ,685 ,730 ,114 ,635 ,680 ,
259 ,687 ,732 ,66 ,494 ,731 ,26 ,308 ,670 ,
27 ,348 ,730 ,29 ,124 ,682 ,16 ,22 ,397 ,
15 ,41 ,516 ,19 ,13 ,275 ,12 ,9 ,72 ,
11 ,16 ,126 ,15 ,7 ,31 ,11 ,6 ,12 ,
9 ,11 ,16 ,13 ,5 ,13 ,11 ,5 ,6 ,
9 ,9 ,10 ,12 ,5 ,9 ,11 ,5 ,5 ,
7 ,7 ,7 ,10 ,3 ,7 ,9 ,4 ,3 ,
7 ,6 ,5 ,9 ,2 ,6 ,8 ,3 ,3 ,
14 ,6 ,4 ,11 ,2 ,6 ,8 ,3 ,2 ,
82 ,6 ,4 ,19 ,2 ,5 ,9 ,2 ,2 ,
345 ,70 ,4 ,301 ,3 ,4 ,328 ,3 ,1 ,
626 ,210 ,75 ,673 ,129 ,5 ,708 ,198 ,2 ,
692 ,564 ,82 ,723 ,533 ,30 ,730 ,574 ,13 ,
692 ,706 ,578 ,724 ,682 ,575 ,732 ,689 ,525 ,
696 ,711 ,728 ,726 ,687 ,726 ,733 ,691 ,671 ,
692 ,709 ,728 ,725 ,686 ,728 ,733 ,692 ,677 ,
690 ,705 ,726 ,715 ,675 ,723 ,694 ,659 ,652 ,
682 ,698 ,724 ,693 ,656 ,714 ,608 ,586 ,598 ,
659 ,681 ,717 ,640 ,609 ,691 ,486 ,478 ,513 ,
606 ,639 ,701 ,547 ,528 ,643 ,316 ,314 ,362 ,
523 ,574 ,668 ,399 ,387 ,542 ,194 ,172 ,197 ,
373 ,440 ,576 ,240 ,217 ,334 ,110 ,88 ,90 ,
219 ,262 ,384 ,131 ,99 ,135 ,46 ,31 ,28 ,
115 ,132 ,152 ,50 ,29 ,43 ,49 ,23 ,15 ,
37 ,43 ,40 ,43 ,22 ,26 ,61 ,36 ,13 ,
31 ,36 ,25 ,55 ,34 ,24 ,82 ,55 ,13 ,
42 ,52 ,23 ,74 ,58 ,24 ,187 ,96 ,20 ,
57 ,77 ,24 ,144 ,94 ,38 ,389 ,163 ,58 ,
118 ,116 ,39 ,300 ,150 ,81 ,577 ,314 ,138 ,
254 ,183 ,79 ,482 ,252 ,149 ,657 ,482 ,279 ,
421 ,284 ,138 ,602 ,407 ,250 ,699 ,583 ,436 ,
545 ,431 ,205 ,670 ,529 ,402 ,721 ,646 ,545 ,
616 ,546 ,322 ,700 ,601 ,531 ,723 ,668 ,612 ,
658 ,617 ,458 ,713 ,649 ,625 ,729 ,681 ,650 ,
677 ,666 ,576 ,720 ,670 ,686 ,732 ,688 ,667 ,
688 ,696 ,663 ,726 ,682 ,716 ,732 ,691 ,674 ,
684 ,683 ,614 ,724 ,678 ,703 ,734 ,692 ,672 ,
667 ,637 ,502 ,716 ,659 ,647 ,730 ,684 ,657 ,
632 ,574 ,373 ,707 ,622 ,568 ,726 ,676 ,629 ,
577 ,482 ,245 ,687 ,562 ,459 ,724 ,661 ,575 ,
488 ,353 ,186 ,640 ,473 ,323 ,711 ,617 ,493 ,
365 ,244 ,144 ,578 ,354 ,222 ,689 ,563 ,391 ,
240 ,166 ,115 ,471 ,229 ,160 ,649 ,470 ,263 ,
142 ,111 ,90 ,332 ,154 ,128 ,577 ,326 ,170 ,
66 ,73 ,55 ,201 ,101 ,97 ,426 ,193 ,107 ,
35 ,46 ,33 ,92 ,62 ,60 ,248 ,111 ,59 ,
25 ,28 ,23 ,46 ,34 ,35 ,101 ,65 ,34 ,
22 ,25 ,22 ,33 ,19 ,25 ,44 ,36 ,20 ,
39 ,46 ,36 ,28 ,15 ,23 ,29 ,20 ,14 ,
126 ,148 ,190 ,40 ,28 ,37 ,25 ,16 ,13 ,
282 ,344 ,483 ,140 ,128 ,182 ,49 ,33 ,26 ,
482 ,535 ,641 ,310 ,312 ,449 ,130 ,105 ,107 ,
604 ,635 ,697 ,526 ,515 ,626 ,265 ,258 ,288 ,
664 ,684 ,718 ,641 ,611 ,689 ,479 ,465 ,489 ,
685 ,701 ,724 ,698 ,661 ,715 ,622 ,594 ,596 ,
691 ,706 ,727 ,715 ,674 ,721 ,698 ,658 ,648 ,
696 ,712 ,733 ,726 ,688 ,731 ,732 ,692 ,676 ,
692 ,708 ,726 ,724 ,685 ,722 ,733 ,692 ,661 ,
694 ,708 ,653 ,726 ,680 ,569 ,732 ,677 ,420 ,
693 ,603 ,517 ,724 ,510 ,289 ,728 ,436 ,84 ,
613 ,457 ,248 ,614 ,263 ,141 ,544 ,137 ,45 ,
434 ,207 ,45 ,330 ,100 ,37 ,213 ,49 ,18 ,
250 ,42 ,28 ,178 ,24 ,32 ,84 ,18 ,15 ,
38 ,28 ,21 ,47 ,18 ,24 ,31 ,17 ,12 ,
26 ,24 ,21 ,35 ,16 ,21 ,27 ,16 ,11 ,
23 ,23 ,20 ,33 ,14 ,21 ,28 ,14 ,11 ,
21 ,23 ,20 ,27 ,13 ,21 ,25 ,13 ,11 ,
23 ,22 ,20 ,26 ,13 ,22 ,24 ,13 ,11 ,
21 ,23 ,25 ,25 ,13 ,22 ,24 ,14 ,11 ,
22 ,26 ,331 ,25 ,14 ,138 ,24 ,14 ,27 ,
25 ,148 ,666 ,28 ,60 ,581 ,26 ,27 ,290 ,
95 ,520 ,720 ,51 ,343 ,706 ,35 ,203 ,593 ,
393 ,677 ,728 ,290 ,618 ,725 ,164 ,529 ,662 ,
632 ,704 ,728 ,604 ,673 ,727 ,485 ,660 ,674 ,
685 ,710 ,731 ,706 ,685 ,730 ,679 ,688 ,676 ,
691 ,709 ,729 ,721 ,683 ,728 ,725 ,689 ,675 ,
693 ,709 ,730 ,724 ,684 ,728 ,729 ,690 ,676 ,
694 ,712 ,732 ,718 ,689 ,732 ,685 ,692 ,679 ,
643 ,709 ,730 ,604 ,681 ,728 ,383 ,670 ,676 ,
327 ,688 ,732 ,180 ,617 ,731 ,67 ,472 ,672 ,
120 ,502 ,728 ,58 ,236 ,710 ,37 ,81 ,578 ,
96 ,127 ,644 ,41 ,32 ,489 ,28 ,24 ,150 ,
33 ,100 ,186 ,31 ,20 ,64 ,25 ,17 ,28 ,
25 ,37 ,65 ,27 ,15 ,32 ,24 ,15 ,17 ,
22 ,28 ,40 ,25 ,13 ,25 ,22 ,13 ,14 ,
20 ,24 ,25 ,24 ,12 ,22 ,22 ,13 ,12 ,
20 ,22 ,22 ,23 ,11 ,20 ,21 ,12 ,11 ,
21 ,21 ,20 ,23 ,11 ,19 ,21 ,12 ,10 ,
164 ,22 ,18 ,151 ,12 ,18 ,81 ,12 ,9 ,
501 ,66 ,18 ,367 ,49 ,18 ,167 ,25 ,9 ,
618 ,520 ,32 ,599 ,316 ,35 ,514 ,112 ,13 ,
686 ,583 ,561 ,716 ,466 ,401 ,716 ,316 ,98 ,
694 ,696 ,634 ,724 ,659 ,468 ,730 ,627 ,152 ,
692 ,706 ,706 ,723 ,680 ,683 ,730 ,684 ,535 ,
695 ,711 ,729 ,726 ,688 ,726 ,733 ,691 ,663 ,
693 ,709 ,729 ,723 ,684 ,726 ,730 ,689 ,671 ,
696 ,712 ,732 ,726 ,688 ,730 ,727 ,686 ,672 ,
690 ,707 ,728 ,716 ,677 ,724 ,687 ,659 ,654 ,
680 ,700 ,725 ,682 ,652 ,714 ,572 ,571 ,593 ,
642 ,675 ,717 ,600 ,586 ,683 ,383 ,404 ,462 ,
557 ,612 ,693 ,436 ,445 ,603 ,206 ,203 ,252 ,
385 ,474 ,619 ,232 ,220 ,380 ,90 ,78 ,92 ,
201 ,260 ,416 ,111 ,87 ,137 ,35 ,24 ,25 ,
101 ,121 ,171 ,44 ,26 ,43 ,28 ,16 ,16 ,
39 ,46 ,49 ,32 ,17 ,26 ,26 ,16 ,13 ,
25 ,27 ,26 ,29 ,15 ,23 ,29 ,25 ,13 ,
23 ,25 ,22 ,30 ,20 ,24 ,63 ,40 ,18 ,
24 ,28 ,22 ,53 ,33 ,36 ,209 ,61 ,26 ,
43 ,42 ,33 ,149 ,54 ,59 ,432 ,132 ,48 ,
124 ,66 ,61 ,341 ,110 ,85 ,604 ,288 ,107 ,
300 ,135 ,91 ,540 ,224 ,128 ,681 ,485 ,219 ,
496 ,282 ,132 ,648 ,428 ,223 ,716 ,605 ,418 ,
608 ,472 ,210 ,699 ,564 ,403 ,724 ,659 ,549 ,
663 ,596 ,352 ,715 ,640 ,562 ,730 ,680 ,628 ,
682 ,665 ,514 ,723 ,671 ,662 ,733 ,689 ,661 ,
690 ,696 ,637 ,723 ,680 ,707 ,730 ,689 ,670 ,
691 ,701 ,688 ,726 ,686 ,722 ,734 ,694 ,678 ,
682 ,674 ,580 ,722 ,672 ,679 ,731 ,688 ,663 ,
657 ,608 ,416 ,713 ,638 ,587 ,728 ,678 ,631 ,
590 ,500 ,253 ,690 ,561 ,445 ,723 ,656 ,557 ,
477 ,342 ,166 ,629 ,440 ,279 ,705 ,595 ,439 ,
310 ,204 ,93 ,526 ,274 ,167 ,666 ,493 ,277 ,
165 ,113 ,46 ,360 ,161 ,97 ,585 ,326 ,155 ,
75 ,60 ,28 ,203 ,90 ,50 ,421 ,185 ,82 ,
35 ,30 ,24 ,92 ,47 ,28 ,220 ,91 ,30 ,
25 ,24 ,24 ,47 ,23 ,25 ,87 ,45 ,17 ,
22 ,23 ,22 ,31 ,15 ,25 ,39 ,23 ,13 ,
23 ,26 ,25 ,27 ,14 ,23 ,26 ,15 ,12 ,
74 ,93 ,119 ,28 ,17 ,27 ,24 ,14 ,11 ,
172 ,229 ,374 ,71 ,62 ,87 ,24 ,15 ,11 ,
351 ,436 ,585 ,178 ,178 ,294 ,57 ,44 ,40 ,
516 ,576 ,671 ,348 ,371 ,530 ,145 ,131 ,154 ,
622 ,657 ,710 ,547 ,541 ,654 ,287 ,294 ,351 ,
667 ,688 ,719 ,644 ,620 ,698 ,479 ,482 ,522 ,
685 ,701 ,725 ,696 ,661 ,716 ,605 ,588 ,603 ,
692 ,709 ,729 ,716 ,678 ,725 ,684 ,653 ,651 ,
695 ,711 ,732 ,725 ,686 ,729 ,731 ,690 ,676 ,
692 ,709 ,728 ,725 ,687 ,728 ,733 ,693 ,675 ,
695 ,710 ,700 ,725 ,685 ,653 ,732 ,686 ,532 ,
692 ,654 ,585 ,725 ,582 ,395 ,731 ,537 ,139 ,
645 ,518 ,357 ,655 ,330 ,233 ,614 ,194 ,75 ,
505 ,303 ,58 ,422 ,164 ,47 ,305 ,76 ,22 ,
315 ,75 ,28 ,224 ,41 ,32 ,111 ,24 ,16 ,
92 ,31 ,22 ,84 ,19 ,27 ,44 ,16 ,13 ,
28 ,24 ,20 ,36 ,18 ,21 ,26 ,17 ,12 ,
24 ,21 ,20 ,37 ,14 ,20 ,29 ,14 ,11 ,
20 ,23 ,18 ,27 ,12 ,20 ,24 ,13 ,10 ,
19 ,20 ,18 ,25 ,12 ,19 ,23 ,13 ,10 ,
18 ,20 ,19 ,23 ,12 ,19 ,22 ,12 ,10 ,
19 ,21 ,171 ,23 ,12 ,74 ,22 ,12 ,13 ,
21 ,54 ,552 ,24 ,16 ,389 ,22 ,13 ,184 ,
28 ,368 ,712 ,26 ,224 ,683 ,25 ,87 ,516 ,
276 ,654 ,728 ,126 ,560 ,724 ,56 ,432 ,656 ,
590 ,702 ,728 ,522 ,670 ,727 ,381 ,649 ,674 ,
685 ,710 ,731 ,700 ,685 ,729 ,657 ,686 ,675 ,
690 ,707 ,729 ,721 ,685 ,729 ,727 ,691 ,678 ,
692 ,709 ,730 ,725 ,687 ,730 ,733 ,694 ,680 ,
696 ,712 ,733 ,721 ,689 ,731 ,695 ,692 ,678 ,
644 ,708 ,729 ,595 ,682 ,729 ,336 ,672 ,679 ,
258 ,686 ,733 ,126 ,594 ,730 ,54 ,391 ,667 ,
45 ,373 ,721 ,50 ,107 ,689 ,32 ,43 ,491 ,
36 ,50 ,536 ,35 ,25 ,265 ,26 ,20 ,53 ,
23 ,35 ,56 ,28 ,16 ,40 ,23 ,15 ,21 ,
24 ,24 ,30 ,25 ,13 ,27 ,22 ,13 ,14 ,
19 ,24 ,22 ,23 ,12 ,21 ,21 ,12 ,12 ,
18 ,21 ,19 ,22 ,11 ,19 ,21 ,12 ,11 ,
19 ,19 ,18 ,22 ,10 ,18 ,20 ,11 ,10 ,
33 ,19 ,16 ,44 ,10 ,18 ,31 ,11 ,9 ,
414 ,22 ,16 ,274 ,12 ,17 ,124 ,11 ,9 ,
468 ,231 ,17 ,342 ,134 ,18 ,199 ,48 ,8 ,
643 ,480 ,77 ,647 ,240 ,64 ,607 ,92 ,20 ,
690 ,567 ,539 ,720 ,455 ,318 ,725 ,323 ,69 ,
693 ,695 ,587 ,722 ,656 ,376 ,728 ,637 ,105 ,
692 ,707 ,698 ,725 ,684 ,672 ,733 ,689 ,520 ,
695 ,711 ,728 ,726 ,686 ,724 ,731 ,689 ,659 ,
691 ,708 ,727 ,724 ,684 ,726 ,732 ,691 ,674 ,
693 ,709 ,730 ,724 ,686 ,729 ,724 ,685 ,672 ,
692 ,708 ,729 ,714 ,675 ,724 ,678 ,650 ,649 ,
678 ,698 ,723 ,680 ,649 ,713 ,556 ,556 ,589 ,
640 ,672 ,715 ,587 ,576 ,680 ,354 ,378 ,449 ,
547 ,607 ,690 ,402 ,419 ,587 ,183 ,188 ,242 ,
366 ,469 ,616 ,220 ,215 ,363 ,70 ,66 ,81 ,
175 ,230 ,396 ,72 ,61 ,109 ,31 ,22 ,22 ,
61 ,91 ,116 ,36 ,20 ,33 ,25 ,15 ,15 ,
28 ,32 ,33 ,30 ,16 ,26 ,26 ,14 ,13 ,
22 ,25 ,25 ,30 ,14 ,25 ,41 ,21 ,12 ,
22 ,23 ,22 ,40 ,24 ,22 ,110 ,35 ,12 ,
31 ,37 ,21 ,93 ,41 ,22 ,326 ,99 ,18 ,
76 ,60 ,21 ,253 ,91 ,33 ,559 ,238 ,61 ,
222 ,114 ,30 ,483 ,193 ,79 ,665 ,456 ,167 ,
416 ,237 ,72 ,610 ,371 ,165 ,706 ,578 ,351 ,
556 ,400 ,144 ,680 ,521 ,313 ,722 ,644 ,501 ,
628 ,533 ,242 ,707 ,603 ,473 ,727 ,671 ,590 ,
666 ,614 ,381 ,716 ,651 ,589 ,730 ,683 ,641 ,
683 ,669 ,525 ,722 ,672 ,668 ,731 ,688 ,661 ,
687 ,694 ,633 ,724 ,682 ,709 ,733 ,693 ,675 ,
691 ,701 ,669 ,727 ,685 ,716 ,733 ,693 ,674 ,
676 ,657 ,529 ,718 ,663 ,653 ,731 ,685 ,654 ,
635 ,570 ,344 ,706 ,610 ,527 ,726 ,670 ,599 ,
535 ,407 ,200 ,660 ,498 ,334 ,716 ,622 ,486 ,
362 ,234 ,114 ,560 ,312 ,185 ,676 ,519 ,304 ,
182 ,129 ,60 ,382 ,171 ,107 ,591 ,339 ,162 ,
74 ,67 ,34 ,205 ,95 ,57 ,420 ,182 ,79 ,
34 ,35 ,25 ,87 ,52 ,34 ,222 ,94 ,33 ,
25 ,25 ,24 ,43 ,25 ,26 ,80 ,52 ,19 ,
23 ,24 ,23 ,33 ,17 ,24 ,40 ,29 ,15 ,
23 ,27 ,27 ,28 ,15 ,23 ,28 ,18 ,13 ,
53 ,68 ,74 ,27 ,14 ,23 ,25 ,15 ,12 ,
126 ,154 ,247 ,37 ,27 ,44 ,24 ,15 ,11 ,
232 ,303 ,477 ,104 ,96 ,160 ,31 ,20 ,17 ,
405 ,487 ,622 ,211 ,226 ,384 ,81 ,65 ,71 ,
548 ,602 ,688 ,399 ,423 ,583 ,160 ,147 ,185 ,
628 ,662 ,710 ,561 ,556 ,664 ,304 ,324 ,385 ,
671 ,692 ,721 ,654 ,630 ,703 ,494 ,500 ,537 ,
686 ,703 ,726 ,701 ,667 ,719 ,628 ,612 ,620 ,
694 ,710 ,731 ,719 ,679 ,725 ,702 ,667 ,656 ,
692 ,708 ,729 ,724 ,684 ,728 ,731 ,691 ,677 ,
694 ,710 ,730 ,726 ,689 ,729 ,734 ,694 ,671 ,
693 ,710 ,643 ,724 ,684 ,569 ,730 ,686 ,345 ,
694 ,631 ,296 ,726 ,547 ,108 ,734 ,450 ,38 ,
646 ,238 ,148 ,652 ,105 ,60 ,593 ,44 ,22 ,
237 ,152 ,35 ,197 ,62 ,30 ,78 ,28 ,16 ,
162 ,37 ,27 ,116 ,21 ,27 ,54 ,17 ,14 ,
37 ,28 ,22 ,44 ,17 ,24 ,31 ,16 ,12 ,
26 ,25 ,21 ,33 ,15 ,23 ,26 ,15 ,12 ,
23 ,23 ,20 ,31 ,14 ,22 ,27 ,14 ,12 ,
21 ,23 ,21 ,28 ,14 ,22 ,25 ,14 ,12 ,
21 ,23 ,21 ,27 ,13 ,22 ,26 ,14 ,12 ,
21 ,24 ,35 ,26 ,14 ,25 ,25 ,14 ,12 ,
23 ,25 ,465 ,27 ,14 ,227 ,26 ,14 ,34 ,
26 ,188 ,701 ,29 ,53 ,652 ,27 ,22 ,415 ,
69 ,595 ,723 ,39 ,447 ,716 ,31 ,220 ,624 ,
453 ,693 ,731 ,264 ,642 ,729 ,118 ,560 ,671 ,
645 ,705 ,728 ,624 ,677 ,727 ,486 ,673 ,675 ,
685 ,710 ,731 ,705 ,687 ,731 ,667 ,690 ,678 ,
691 ,710 ,730 ,720 ,684 ,728 ,724 ,688 ,675 ,
692 ,708 ,729 ,724 ,685 ,728 ,731 ,692 ,677 ,
694 ,711 ,731 ,722 ,688 ,731 ,708 ,692 ,678 ,
656 ,709 ,729 ,633 ,681 ,729 ,463 ,669 ,676 ,
406 ,678 ,729 ,339 ,592 ,727 ,195 ,454 ,661 ,
154 ,471 ,713 ,88 ,342 ,669 ,46 ,182 ,489 ,
69 ,140 ,565 ,46 ,42 ,462 ,31 ,29 ,178 ,
31 ,54 ,118 ,34 ,21 ,58 ,26 ,18 ,28 ,
30 ,33 ,46 ,29 ,16 ,34 ,24 ,15 ,19 ,
24 ,28 ,28 ,26 ,13 ,25 ,23 ,13 ,15 ,
20 ,24 ,23 ,24 ,12 ,22 ,21 ,13 ,12 ,
20 ,21 ,20 ,25 ,11 ,20 ,22 ,12 ,11 ,
125 ,21 ,18 ,83 ,11 ,19 ,47 ,12 ,10 ,
431 ,66 ,18 ,271 ,33 ,18 ,149 ,18 ,10 ,
554 ,409 ,21 ,519 ,185 ,22 ,446 ,64 ,10 ,
663 ,517 ,400 ,678 ,367 ,206 ,666 ,228 ,37 ,
690 ,641 ,559 ,720 ,575 ,366 ,726 ,521 ,113 ,
692 ,704 ,654 ,723 ,678 ,564 ,730 ,676 ,377 ,
694 ,709 ,724 ,725 ,685 ,716 ,731 ,688 ,640 ,
692 ,708 ,727 ,723 ,683 ,725 ,731 ,689 ,671 ,
695 ,711 ,731 ,727 ,688 ,730 ,733 ,693 ,677 ,
683 ,699 ,724 ,708 ,669 ,719 ,689 ,652 ,644 ,
671 ,689 ,718 ,692 ,653 ,711 ,655 ,625 ,623 ,
656 ,678 ,717 ,672 ,636 ,705 ,613 ,588 ,596 ,
623 ,649 ,704 ,628 ,594 ,683 ,541 ,527 ,547 ,
580 ,615 ,690 ,579 ,554 ,657 ,459 ,450 ,483 ,
520 ,566 ,662 ,497 ,477 ,605 ,364 ,354 ,385 ,
439 ,495 ,617 ,411 ,388 ,526 ,309 ,282 ,300 ,
353 ,407 ,534 ,334 ,299 ,410 ,319 ,254 ,237 ,
285 ,318 ,420 ,321 ,261 ,308 ,372 ,260 ,214 ,
282 ,291 ,320 ,346 ,248 ,248 ,437 ,276 ,202 ,
305 ,295 ,251 ,382 ,250 ,231 ,519 ,350 ,242 ,
353 ,320 ,235 ,451 ,305 ,253 ,593 ,436 ,317 ,
408 ,363 ,241 ,524 ,380 ,305 ,635 ,509 ,396 ,
478 ,434 ,281 ,583 ,456 ,379 ,668 ,561 ,464 ,
538 ,503 ,342 ,623 ,509 ,446 ,686 ,593 ,514 ,
582 ,555 ,408 ,658 ,558 ,522 ,705 ,631 ,564 ,
621 ,600 ,487 ,681 ,598 ,578 ,712 ,649 ,598 ,
642 ,632 ,550 ,697 ,630 ,628 ,721 ,668 ,632 ,
665 ,664 ,613 ,709 ,654 ,671 ,723 ,675 ,648 ,
673 ,679 ,658 ,713 ,665 ,696 ,726 ,681 ,661 ,

};
const  unsigned  char  testlabel[]  PROGMEM  =  {3.0 ,2.0 ,1.0 ,4.0 ,3.0 ,2.0 ,1.0,4.0 ,3.0 ,2.0 ,1.0 ,4.0 ,3.0 ,2.0 ,1.0 ,4.0 ,3.0 ,2.0 ,1.0 ,4.0 ,3.0 ,2.0 ,1.0 ,4.0 ,3.0 ,2.0 ,1.0 ,4.0 ,3.0 ,2.0 ,1.0 ,4.0 };
#endif
