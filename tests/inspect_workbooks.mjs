import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const files = process.argv.slice(2);
for (const file of files) {
  const input = await FileBlob.load(file);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheets = JSON.parse((await workbook.inspect({
    kind: "sheet",
    include: "id,name",
    maxChars: 20000,
  })).ndjson.split("\n").filter(Boolean).map(JSON.parse).map(x => x.data ?? x));
  process.stdout.write(`\n### ${path.basename(file)}\n`);
  process.stdout.write(JSON.stringify(sheets, null, 2) + "\n");
  const summary = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 30000,
    tableMaxRows: 8,
    tableMaxCols: 12,
    tableMaxCellChars: 120,
  });
  process.stdout.write(summary.ndjson + "\n");
}
