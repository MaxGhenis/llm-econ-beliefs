import { NextRequest, NextResponse } from "next/server";

import { loadRunPayload } from "@/lib/dashboard-data";

export async function GET(request: NextRequest) {
  const quantityId = request.nextUrl.searchParams.get("quantityId");
  const modelName = request.nextUrl.searchParams.get("modelName");

  if (!quantityId || !modelName) {
    return NextResponse.json(
      { error: "quantityId and modelName are required" },
      { status: 400 },
    );
  }

  const payload = loadRunPayload(quantityId, modelName);
  if (!payload) {
    return NextResponse.json(
      { error: "No matching runs found" },
      { status: 404 },
    );
  }

  return NextResponse.json(payload);
}
