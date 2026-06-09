export type PlanTier = "free" | "basic" | "pro";

export interface PlanLimits {
  tier: PlanTier;
  maxProjects: number;
  maxWordsPerProject: number;
  canFix: boolean;
  fixesPerMonth: number;
  stripePriceEnv?: "stripePriceBasic" | "stripePricePro";
}

export const PLANS: Record<PlanTier, PlanLimits> = {
  free: { tier: "free", maxProjects: 1, maxWordsPerProject: 3_000, canFix: false, fixesPerMonth: 0 },
  basic: { tier: "basic", maxProjects: 20, maxWordsPerProject: 300_000, canFix: false, fixesPerMonth: 0, stripePriceEnv: "stripePriceBasic" },
  pro: { tier: "pro", maxProjects: 200, maxWordsPerProject: 2_000_000, canFix: true, fixesPerMonth: 50, stripePriceEnv: "stripePricePro" },
};

export const limitsFor = (tier: PlanTier): PlanLimits => PLANS[tier];
