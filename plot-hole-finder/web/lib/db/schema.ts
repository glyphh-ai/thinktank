// Postgres schema (Drizzle). The durable backbone: users/auth, billing, projects,
// documents, the entity-indexed memory store, issues+evidence, timeline, notes,
// and the job queue.

import { sql } from "drizzle-orm";
import {
  boolean,
  doublePrecision,
  index,
  integer,
  jsonb,
  pgTable,
  text,
  timestamp,
  uniqueIndex,
} from "drizzle-orm/pg-core";

const id = () => text("id").primaryKey().$defaultFn(() => crypto.randomUUID());
const createdAt = () => timestamp("created_at", { withTimezone: true }).defaultNow().notNull();

export const users = pgTable("users", {
  id: id(),
  email: text("email").notNull(),
  emailVerified: boolean("email_verified").default(false).notNull(),
  passwordHash: text("password_hash").notNull(),
  status: text("status").default("active").notNull(), // active | suspended
  stripeCustomerId: text("stripe_customer_id"),
  createdAt: createdAt(),
}, (t) => ({ emailIdx: uniqueIndex("users_email_idx").on(t.email) }));

export const subscriptions = pgTable("subscriptions", {
  id: id(),
  userId: text("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  plan: text("plan").default("free").notNull(), // free | basic | pro
  status: text("status").default("active").notNull(),
  stripeSubscriptionId: text("stripe_subscription_id"),
  stripePriceId: text("stripe_price_id"),
  currentPeriodEnd: timestamp("current_period_end", { withTimezone: true }),
  cancelAtPeriodEnd: boolean("cancel_at_period_end").default(false).notNull(),
  createdAt: createdAt(),
}, (t) => ({ userIdx: index("subs_user_idx").on(t.userId) }));

export const verificationTokens = pgTable("verification_tokens", {
  id: id(),
  userId: text("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  kind: text("kind").notNull(), // email_pin | password_reset
  secretHash: text("secret_hash").notNull(),
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
  consumedAt: timestamp("consumed_at", { withTimezone: true }),
  attempts: integer("attempts").default(0).notNull(),
  createdAt: createdAt(),
}, (t) => ({ userIdx: index("vtok_user_idx").on(t.userId) }));

export const refreshTokens = pgTable("refresh_tokens", {
  id: id(),
  userId: text("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  tokenHash: text("token_hash").notNull(),
  expiresAt: timestamp("expires_at", { withTimezone: true }).notNull(),
  revokedAt: timestamp("revoked_at", { withTimezone: true }),
  createdAt: createdAt(),
}, (t) => ({ hashIdx: uniqueIndex("rtok_hash_idx").on(t.tokenHash) }));

export const projects = pgTable("projects", {
  id: id(),
  userId: text("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  status: text("status").default("created").notNull(),
  wordCount: integer("word_count").default(0).notNull(),
  error: text("error"),
  createdAt: createdAt(),
}, (t) => ({ userIdx: index("projects_user_idx").on(t.userId) }));

// One document per project for now (content stored inline; blob/R2 later).
export const documents = pgTable("documents", {
  id: id(),
  projectId: text("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  filename: text("filename").notNull(),
  mimeType: text("mime_type").default("text/plain").notNull(),
  content: text("content").notNull(),
  byteSize: integer("byte_size").default(0).notNull(),
  wordCount: integer("word_count").default(0).notNull(),
  sha256: text("sha256"),
  createdAt: createdAt(),
}, (t) => ({ projectIdx: uniqueIndex("documents_project_idx").on(t.projectId) }));

export const memoryFacts = pgTable("memory_facts", {
  id: id(),
  projectId: text("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  entity: text("entity").notNull(),
  key: text("key").notNull(),
  value: text("value").notNull(),
  kind: text("kind").default("state").notNull(), // state | event
  chunkIndex: integer("chunk_index").default(0).notNull(),
  storyTime: integer("story_time"),
  charStart: integer("char_start"),
  createdAt: createdAt(),
}, (t) => ({ entityIdx: index("memory_entity_idx").on(t.projectId, t.entity) }));

export const issues = pgTable("issues", {
  id: id(),
  projectId: text("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  category: text("category").notNull(),
  severity: text("severity").default("medium").notNull(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  suggestedFix: text("suggested_fix"),
  status: text("status").default("open").notNull(),
  createdAt: createdAt(),
}, (t) => ({ projectIdx: index("issues_project_idx").on(t.projectId) }));

export const issueEvidence = pgTable("issue_evidence", {
  id: id(),
  issueId: text("issue_id").notNull().references(() => issues.id, { onDelete: "cascade" }),
  role: text("role").default("context").notNull(),
  charStart: integer("char_start").notNull(),
  charEnd: integer("char_end").notNull(),
  quote: text("quote").notNull(),
});

export const timelineEvents = pgTable("timeline_events", {
  id: id(),
  projectId: text("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  summary: text("summary").notNull(),
  entity: text("entity"),
  narrativeTime: integer("narrative_time").notNull(),
  storyTime: integer("story_time").notNull(),
  charStart: integer("char_start").notNull(),
  charEnd: integer("char_end").notNull(),
}, (t) => ({ projectIdx: index("timeline_project_idx").on(t.projectId) }));

export const notes = pgTable("notes", {
  id: id(),
  projectId: text("project_id").notNull().references(() => projects.id, { onDelete: "cascade" }),
  charStart: integer("char_start").notNull(),
  charEnd: integer("char_end").notNull(),
  text: text("text").notNull(),
  kind: text("kind").default("note").notNull(),
  issueId: text("issue_id"),
}, (t) => ({ projectIdx: index("notes_project_idx").on(t.projectId) }));

export const jobs = pgTable("jobs", {
  id: id(),
  userId: text("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  projectId: text("project_id").references(() => projects.id, { onDelete: "cascade" }),
  type: text("type").notNull(), // analyze | fix
  status: text("status").default("queued").notNull(),
  payload: jsonb("payload").default({}).notNull(),
  result: jsonb("result"),
  error: text("error"),
  progress: doublePrecision("progress").default(0).notNull(),
  step: text("step"),
  attempts: integer("attempts").default(0).notNull(),
  maxAttempts: integer("max_attempts").default(3).notNull(),
  scheduledAt: timestamp("scheduled_at", { withTimezone: true }).defaultNow().notNull(),
  leaseExpiresAt: timestamp("lease_expires_at", { withTimezone: true }),
  createdAt: createdAt(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull().$onUpdate(() => sql`now()`),
}, (t) => ({ statusIdx: index("jobs_status_idx").on(t.status, t.scheduledAt) }));
