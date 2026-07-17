# Semantic diff: Qwen2.5-0.5B → Qwen2.5-0.5B-Instruct

1375 words x 242 named axes, layer 12. Each model measured in its own frame (axes rebuilt from its own seed vectors); readings z-scored within model; diff = z_inst − z_base.

## Axis stability under instruction tuning
median axis correlation base↔instruct: **0.993**

most changed axes | most stable axes
---|---
optimist-pessimist (0.971) | patient-impatient (0.997)
inhale-exhale (0.979) | profit-loss (0.996)
prince-princess (0.980) | single-married (0.996)
honest-dishonest (0.981) | order-chaos (0.996)
inside-outside (0.984) | long-moment (0.996)
arrival-departure (0.985) | create-ruin (0.996)
tie-untie (0.985) | hire-fire (0.996)
sunrise-sunset (0.986) | open-shut (0.996)

## Most semantically moved words (||Δz|| over 242 axes)

**mean** (4.0), **issue** (2.6), **date** (2.5), **reply** (2.5), **messy** (2.4), **pretty** (2.4), **cruel** (2.4), **casino** (2.4), **treacherous** (2.4), **shadow** (2.4), **dim** (2.3), **interest** (2.3), **despair** (2.3), **answer** (2.3), **courage** (2.3), **lie** (2.3), **lucky** (2.3), **shallow** (2.3), **weather** (2.3), **exhale** (2.3)

## Targeted changelog

**good-bad** (axis corr 0.99): toward *good*: updated (+0.3), update (+0.3), edit (+0.3), useful (+0.3), services (+0.3), help (+0.3)
  toward *bad*: mean (-0.6), off (-0.4), online (-0.3), innocent (-0.3), evening (-0.3), some (-0.3)

**polite-rude** (axis corr 0.99): toward *polite*: updated (+0.4), update (+0.4), june (+0.3), save (+0.3), press (+0.3), july (+0.3)
  toward *rude*: mean (-0.4), selfish (-0.4), tenant (-0.4), casino (-0.4), dishonest (-0.3), hostile (-0.3)

**safe-dangerous** (axis corr 0.99): toward *safe*: gentle (+0.3), class (+0.3), domestic (+0.3), famous (+0.2), departure (+0.2), remember (+0.2)
  toward *dangerous*: mean (-0.4), casino (-0.3), usr (-0.3), quality (-0.3), fail (-0.3), soul (-0.3)

**formal-casual** (axis corr 0.99): toward *formal*: minutes (+0.4), advertise (+0.4), company (+0.3), required (+0.3), cry (+0.3), cause (+0.3)
  toward *casual*: wild (-0.4), further (-0.3), new (-0.3), out (-0.3), while (-0.3), back (-0.3)

**obey-rebel** (axis corr 0.99): toward *obey*: listing (+0.3), going (+0.3), select (+0.3), list (+0.3), left (+0.3), regional (+0.3)
  toward *rebel*: weakness (-0.3), sorrow (-0.3), crazy (-0.3), mean (-0.3), drive (-0.2), hate (-0.2)

**true-false** (axis corr 0.99): toward *true*: auto (+0.5), feb (+0.4), hand (+0.4), hidden (+0.3), self (+0.3), personal (+0.3)
  toward *false*: mean (-0.5), cheap (-0.4), tragedy (-0.4), poverty (-0.4), cruel (-0.4), strong (-0.4)

**kind-cruel** (axis corr 0.99): toward *kind*: difficult (+0.3), believe (+0.3), truth (+0.2), courage (+0.2), remember (+0.2), care (+0.2)
  toward *cruel*: casino (-0.4), mean (-0.3), sony (-0.3), date (-0.3), website (-0.2), clinic (-0.2)

**calm-angry** (axis corr 0.99): toward *calm*: expand (+0.3), reply (+0.3), nature (+0.3), natural (+0.3), exchange (+0.3), changes (+0.3)
  toward *angry*: mean (-0.6), prince (-0.4), law (-0.3), though (-0.3), logic (-0.3), lie (-0.3)

**friendly-hostile** (axis corr 0.99): toward *friendly*: prey (+0.4), sister (+0.4), students (+0.3), teacher (+0.3), employee (+0.3), teach (+0.3)
  toward *hostile*: finish (-0.4), mean (-0.4), yahoo (-0.4), london (-0.4), formal (-0.4), rarely (-0.4)

**honest-dishonest** (axis corr 0.98): toward *honest*: additional (+0.5), effect (+0.5), effects (+0.5), changes (+0.5), discount (+0.4), user (+0.4)
  toward *dishonest*: rude (-0.6), mean (-0.6), gay (-0.5), room (-0.5), cruel (-0.5), brother (-0.5)

## Committed bit flips (|z| >= 1 in both models, sign reversed)
0 flips across 1375x242 bits (0.00%)

