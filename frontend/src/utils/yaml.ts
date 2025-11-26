import { FlowDocument } from "../types";
import { stringify } from "yaml";

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const shouldDrop = (value: unknown) => {
  if (value === undefined || value === null) return true;
  if (typeof value === "string") return value.trim() === "";
  if (Array.isArray(value)) return value.length === 0;
  if (isPlainObject(value)) return Object.keys(value).length === 0;
  return false;
};

const deepClean = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    const cleaned = value
      .map((item) => deepClean(item))
      .filter((item) => !shouldDrop(item));
    return cleaned;
  }

  if (isPlainObject(value)) {
    const entries = Object.entries(value)
      .map(([key, val]) => [key, deepClean(val)])
      .filter(([, val]) => !shouldDrop(val));

    return entries.reduce<Record<string, unknown>>((acc, [key, val]) => {
      acc[key] = val;
      return acc;
    }, {});
  }

  return value;
};

export const normalizeDocument = (doc: FlowDocument): FlowDocument => {
  const version = doc.version?.trim() || "0.1";
  const cleaned = deepClean({ ...doc, version }) as FlowDocument;
  return cleaned;
};

export const toYaml = (doc: FlowDocument) => stringify(normalizeDocument(doc), { lineWidth: 92 });
