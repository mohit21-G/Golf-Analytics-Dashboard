import streamlit as st
import pandas as pd
from datetime import datetime
import calendar

st.set_page_config(page_title="Golf Rate Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------

def convert_drive_link(link):
    file_id = link.split("/d/")[1].split("/")[0]
    return f"https://drive.google.com/uc?id={file_id}"

DATASET = convert_drive_link("https://drive.google.com/file/d/1u0-AbwITOPix3_x-8wWTrWCI-IMl6tIz/view?usp=sharing")

@st.cache_data
def load_data():
    df = pd.read_csv(DATASET)
    df.columns = df.columns.str.strip().str.lower()
    df['tee_date'] = pd.to_datetime(df['tee_date'])
    return df

df = load_data()

if 'tee_time' in df.columns:
    df['hour'] = pd.to_datetime(df['tee_time'], errors='coerce').dt.hour


def _analytics_day_data(selected_date):
  return df[df['tee_date'].dt.date == selected_date]


def _details_day_data(selected_date, selected_course):
  return df[
    (df['course_name'] == selected_course) &
    (df['tee_date'].dt.date == selected_date)
  ]


def _dialog_css():
  # Inject CSS into the parent Streamlit page via JS so it can override
  # Streamlit's own dialog sizing (width="large" gives us ~900px baseline,
  # then we push it wider and remove the vertical scrollbar).
  return """
    <script>
    (function() {
      var id = 'gc-dialog-style';
      if (document.getElementById(id)) return;
      var s = document.createElement('style');
      s.id = id;
      s.textContent = `
        /* ── Widen and fix the dialog shell ── */
        div[data-testid="stDialog"] {
          width: min(78vw, 1160px) !important;
          max-width: 78vw !important;
          left: 50% !important;
          top: 50% !important;
          transform: translate(-50%, -50%) !important;
          position: fixed !important;
          margin: 0 !important;
        }
        div[data-testid="stDialog"] > div {
          width: 100% !important;
          max-height: 90vh !important;
          border-radius: 16px !important;
          overflow: hidden !important;
        }
        /* Remove vertical scrollbar — let content breathe */
        div[data-testid="stDialog"] [data-testid="stVerticalBlock"] {
          overflow-y: visible !important;
          overflow-x: hidden !important;
          max-height: none !important;
          padding: 0.5rem 1.1rem 1rem 1.1rem !important;
        }
        /* Tighten header */
        div[data-testid="stDialog"] h1,
        div[data-testid="stDialog"] h2,
        div[data-testid="stDialog"] h3 {
          margin-top: 0.2rem !important;
          margin-bottom: 0.35rem !important;
          padding-top: 0 !important;
        }
        div[data-testid="stDialog"] [data-testid="stVerticalBlock"] > div:first-child {
          padding-top: 0 !important;
          margin-top: 0 !important;
        }
        /* Plotly — flush, no extra gap */
        div[data-testid="stDialog"] .stPlotlyChart {
          margin-top: 0 !important;
          margin-bottom: 0 !important;
        }
        /* Dataframe — horizontal scroll only */
        div[data-testid="stDialog"] div[data-testid="stDataFrame"] {
          overflow-x: auto !important;
          width: 100% !important;
        }
        /* Selectbox compact */
        div[data-testid="stDialog"] div[data-testid="stSelectbox"] {
          margin-bottom: 0.4rem !important;
        }
        /* Close button compact */
        div[data-testid="stDialog"] div[data-testid="stButton"] {
          margin-top: 0.5rem !important;
        }
        /* Blur backdrop */
        dialog::backdrop {
          background: rgba(2, 6, 23, 0.68) !important;
          backdrop-filter: blur(8px) !important;
          -webkit-backdrop-filter: blur(8px) !important;
        }
        @media (max-width: 1200px) {
          div[data-testid="stDialog"] {
            width: min(86vw, 1000px) !important;
            max-width: 86vw !important;
          }
        }
        @media (max-width: 800px) {
          div[data-testid="stDialog"] {
            width: 96vw !important;
            max-width: 96vw !important;
          }
          div[data-testid="stDialog"] > div {
            max-height: 92vh !important;
          }
          div[data-testid="stDialog"] [data-testid="stVerticalBlock"] {
            padding: 0.4rem 0.6rem 0.6rem 0.6rem !important;
          }
        }
      `;
      // Inject into parent page so it applies to Streamlit's outer DOM
      var target = window.parent ? window.parent.document.head : document.head;
      target.appendChild(s);
    })();
    </script>
  """


@st.dialog("📊 Analytics", width="large", dismissible=True)
def show_analytics_dialog(selected_date, selected_course):
  st.markdown(_dialog_css(), unsafe_allow_html=True)
  pretty_course = str(selected_course).replace("_", " ").strip().title()
  st.header(f"📊 Analytics  ·  {selected_date}  ·  {pretty_course}")
  day_data = _analytics_day_data(selected_date)

  if day_data.empty:
    st.warning("No data available for this date")
  else:
    view_options = ["Course-wise Price", "Hour-wise Price", "Channel-wise Price"]
    view_type = st.selectbox(
      "Select Analytics View",
      view_options,
      index=0,
      key=f"modal_analytics_view_{selected_date}"
    )

    if view_type == "Course-wise Price":
      course_df = (
        df[df['tee_date'].dt.date == selected_date]
        .groupby("course_name")["avg_price"]
        .mean()
        .reset_index()
        .sort_values("avg_price", ascending=False)
      )
      if course_df.empty:
        st.warning("No course data available for this date.")
      else:
        import plotly.graph_objects as go
        colors = []
        avg_price = course_df["avg_price"].mean()
        for c, price in zip(course_df["course_name"], course_df["avg_price"]):
          if c == selected_course:
            colors.append("#f97316")
          elif price > avg_price:
            colors.append("#ef4444")
          else:
            colors.append("#22c55e")
        # Dynamically size chart height based on number of courses so it fits
        # inside the popup without requiring vertical scroll.
        n_courses = len(course_df)
        chart_height = max(320, min(460, 260 + n_courses * 18))
        fig = go.Figure(go.Bar(
          x=course_df["course_name"],
          y=course_df["avg_price"],
          marker_color=colors,
          text=[f"${v:.0f}" for v in course_df["avg_price"]],
          textposition="outside",
        ))
        fig.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font=dict(color="white", size=12),
          xaxis=dict(
            title="Course Name",
            tickangle=-28,
            tickfont=dict(size=11),
            automargin=True,
          ),
          yaxis=dict(title="Avg Price ($)", automargin=True),
          height=chart_height,
          margin=dict(t=24, b=80, l=48, r=16),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view_type == "Hour-wise Price":
      if "hour" not in day_data.columns:
        st.error("No hour data found (need 'tee_time' column)")
      else:
        hour_df = day_data.groupby("hour")["avg_price"].mean().reset_index()
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(
          x=hour_df["hour"],
          y=hour_df["avg_price"],
          mode="lines+markers",
          line=dict(color="#22c55e", width=2),
          marker=dict(size=7, color="#22c55e"),
          text=[f"${v:.0f}" for v in hour_df["avg_price"]],
          hovertemplate="Hour %{x}: $%{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font=dict(color="white", size=12),
          xaxis=dict(title="Hour of Day", dtick=1, automargin=True),
          yaxis=dict(title="Avg Price ($)", automargin=True),
          height=360,
          margin=dict(t=24, b=48, l=48, r=16),
          title=dict(text="⏰ Price by Hour", font=dict(size=14), x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view_type == "Channel-wise Price":
      if "source_channel" in day_data.columns:
        ch_df = day_data.groupby("source_channel")["avg_price"].mean().reset_index()
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
          x=ch_df["source_channel"],
          y=ch_df["avg_price"],
          marker_color="#3b82f6",
          text=[f"${v:.0f}" for v in ch_df["avg_price"]],
          textposition="outside",
        ))
        fig.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font=dict(color="white", size=12),
          xaxis=dict(title="Channel", automargin=True),
          yaxis=dict(title="Avg Price ($)", automargin=True),
          height=360,
          margin=dict(t=24, b=56, l=48, r=16),
          title=dict(text="📡 Price by Channel", font=dict(size=14), x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
      else:
        st.warning("No source_channel column found")

  if st.button("Close", key=f"close_analytics_{selected_date}"):
    if "analytics_dialog_date" in st.session_state:
      del st.session_state["analytics_dialog_date"]
    st.rerun()


@st.dialog("📅 Details", width="large", dismissible=True)
def show_details_dialog(selected_date, selected_course):
  st.markdown(_dialog_css(), unsafe_allow_html=True)
  pretty_course = str(selected_course).replace("_", " ").strip().title()
  st.header(f"📅 {pretty_course}  ·  {selected_date}")
  day_data = _details_day_data(selected_date, selected_course)
  if day_data.empty:
    st.warning("No data for this course and date")
  else:
    detail_df = day_data.copy()

    # ── Format tee_date as plain date string ──────────────────────────────────
    if "tee_date" in detail_df.columns:
      detail_df["tee_date"] = pd.to_datetime(detail_df["tee_date"], errors="coerce").dt.date.astype(str)

    # ── Format price columns as $XX.XX ────────────────────────────────────────
    _PRICE_COLS = {
      "avg_price", "average_price", "min_price", "max_price",
      "market_avg", "market_min", "market_max",
      "market_min_price", "market_avg_price", "market_max_price",
      "brand_current_price", "golfnow_current_price",
      "teeoff_current_price", "supremegolf_current_price",
      "avg_price_avg", "revenue_efficiency_score",
    }
    for col in detail_df.columns:
      if col in _PRICE_COLS:
        detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").map(
          lambda v: f"${v:.2f}" if pd.notna(v) else ""
        )

    # ── Format percentage columns ─────────────────────────────────────────────
    _PCT_COLS = {"occ_percent", "occupancy_percent", "price_gap_percent"}
    for col in detail_df.columns:
      if col in _PCT_COLS:
        detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").map(
          lambda v: f"{v:.1f}%" if pd.notna(v) else ""
        )

    # ── Drop internal/redundant columns and fully-empty columns ──────────────
    # Prefer the richer Drive-dataset columns over the duplicated local ones
    # when both exist (e.g. "average_price" is richer than "avg_price").
    _SKIP_COLS = {"row", "hour", "year", "month", "avg_price_avg"}
    # If the richer column exists, skip the simpler duplicate
    if "average_price" in detail_df.columns:
      _SKIP_COLS.add("avg_price")
    if "occupancy_percent" in detail_df.columns:
      _SKIP_COLS.add("occ_percent")
    if "market_avg_price" in detail_df.columns:
      _SKIP_COLS.update({"market_avg", "market_min", "market_max"})

    display_cols = [
      c for c in detail_df.columns
      if c not in _SKIP_COLS
      and detail_df[c].replace("", pd.NA).dropna().shape[0] > 0
    ]

    detail_view = detail_df[display_cols].reset_index(drop=True)

    # ── Rename to readable labels (no duplicates) ─────────────────────────────
    _RENAME = {
      "course_name":                  "Course",
      "tee_date":                     "Date",
      "tee_time":                     "Tee Time",
      "source_channel":               "Channel",
      "average_price":                "Avg Price",
      "avg_price":                    "Avg Price",
      "min_price":                    "Min Price",
      "max_price":                    "Max Price",
      "occupancy_percent":            "Occupancy %",
      "occ_percent":                  "Occupancy %",
      "total_slots":                  "Total Slots",
      "occupied_slots":               "Occupied Slots",
      "avg_minutes_available":        "Avg Mins Available",
      "market_avg_price":             "Market Avg",
      "market_min_price":             "Market Min",
      "market_max_price":             "Market Max",
      "market_avg":                   "Market Avg",
      "market_min":                   "Market Min",
      "market_max":                   "Market Max",
      "price_gap_percent":            "Price Gap %",
      "price_position_flag":          "Price Position",
      "demand_pressure":              "Demand",
      "revenue_efficiency_score":     "Rev. Efficiency",
      "as_of_date":                   "As Of Date",
      "brand_current_price":          "Brand Price",
      "golfnow_current_price":        "GolfNow Price",
      "teeoff_current_price":         "TeeOff Price",
      "supremegolf_current_price":    "SupremeGolf Price",
      "brand_availability_status":    "Brand Status",
      "golfnow_availability_status":  "GolfNow Status",
      "teeoff_availability_status":   "TeeOff Status",
      "supremegolf_availability_status": "SupremeGolf Status",
      "overall_availability_status":  "Overall Status",
    }
    detail_view = detail_view.rename(columns={k: v for k, v in _RENAME.items() if k in detail_view.columns})

    # Deduplicate any remaining duplicate column names by appending a suffix
    seen: dict[str, int] = {}
    new_cols = []
    for col in detail_view.columns:
      if col in seen:
        seen[col] += 1
        new_cols.append(f"{col} ({seen[col]})")
      else:
        seen[col] = 0
        new_cols.append(col)
    detail_view.columns = new_cols

    row_count = len(detail_view)
    table_height = max(200, min(620, 80 + (row_count * 38)))
    st.dataframe(detail_view, height=table_height, use_container_width=True)
    st.caption(f"{row_count} row{'s' if row_count != 1 else ''} · {selected_course} · {selected_date}")

  if st.button("Close", key=f"close_details_{selected_date}"):
    if "details_dialog_date" in st.session_state:
      del st.session_state["details_dialog_date"]
    st.rerun()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("⚙️ Controls")

courses = df['course_name'].unique()
selected_course = st.sidebar.selectbox("Select Your Course", courses)

df['year'] = df['tee_date'].dt.year
df['month'] = df['tee_date'].dt.month

years = sorted(df['year'].unique())
selected_year = st.sidebar.selectbox("Year", years)

months = sorted(df[df['year'] == selected_year]['month'].unique())
selected_month = st.sidebar.selectbox("Month", months)

# -----------------------------
# FILTER
# -----------------------------

filtered_df = df[
    (df['course_name'] == selected_course) &
    (df['year'] == selected_year) &
    (df['month'] == selected_month)
]

# -----------------------------
# GROUP
# -----------------------------

daily_df = filtered_df.groupby(filtered_df['tee_date'].dt.date).agg({
    "avg_price": "mean",
    "occ_percent": "mean",
    "market_avg": "mean"
}).reset_index()

daily_df.columns = ["date", "price", "occ", "market"]

# -----------------------------
# STYLE
# -----------------------------

st.markdown("""
<style>
html, body, .stApp {
  width: 100%;
  max-width: 100vw;
  overflow-x: hidden !important;
}

.main .block-container {
  padding-top: 1rem;
  padding-left: 1rem;
  padding-right: 1rem;
  padding-bottom: 5rem;
  max-width: 100%;
}

.stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
  overflow-wrap: anywhere;
}

div[data-testid="column"] {
  min-width: 0;
}

div[data-testid="stButton"] > button {
  width: 100%;
  max-width: 100%;
  white-space: normal;
}

div[data-testid="stDataFrame"] {
  width: 100%;
  overflow-x: auto;
}

.stPlotlyChart {
  max-width: 100%;
}

.tile {
  width: 100%;
  min-width: 0;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}
.price {
    font-size: 32px;
    font-weight: bold;
}
.day {
    font-size: 14px;
    margin-bottom: 5px;
}
.badge {
    background-color: rgba(255,255,255,0.25);
    padding: 6px 10px;
    border-radius: 10px;
    font-size: 12px;
    margin-top: 5px;
}

@media (max-width: 1200px) {
  .tile {
    padding: 16px;
  }

  .price {
    font-size: 26px;
  }
}

@media (max-width: 900px) {
  .main .block-container {
    padding-left: 0.75rem;
    padding-right: 0.75rem;
  }

  .tile {
    padding: 14px;
    border-radius: 16px;
  }

  .price {
    font-size: 22px;
  }

  .day {
    font-size: 13px;
  }

  .badge {
    padding: 5px 9px;
    font-size: 11px;
  }
}

@media (max-width: 640px) {
  .main .block-container {
    padding-left: 0.5rem;
    padding-right: 0.5rem;
    padding-bottom: 6rem;
  }

  .tile {
    padding: 10px;
    border-radius: 14px;
  }

  .price {
    font-size: 18px;
  }

  .day {
    font-size: 12px;
  }

  .badge {
    padding: 4px 8px;
    font-size: 10px;
  }

  div[data-testid="stButton"] > button {
    font-size: 0.82rem;
    padding-left: 0.55rem;
    padding-right: 0.55rem;
  }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------

st.title("🏌️ Golf Rate Calendar")
st.write(f"### {selected_course} — {calendar.month_name[selected_month]} {selected_year}")

# -----------------------------
# CALENDAR
# -----------------------------

cal = calendar.monthcalendar(selected_year, selected_month)
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
cols = st.columns(7)

for i, d in enumerate(days):
    cols[i].markdown(f"**{d}**")

for week in cal:
    cols = st.columns(7)

    for i, day in enumerate(week):
        if day == 0:
            cols[i].empty()
        else:
            date = datetime(selected_year, selected_month, day).date()
            row = daily_df[daily_df["date"] == date]

            if not row.empty:
                price = row["price"].values[0]
                occ = row["occ"].values[0]
                market = row["market"].values[0]

                diff = ((price - market) / market) * 100 if market > 0 else 0

                if diff > 10:
                    color = "#ef4444"
                elif diff < -10:
                    color = "#22c55e"
                else:
                    color = "#f59e0b"

                diff_text = f"{diff:+.1f}%"

                cols[i].markdown(f"""
                <div class="tile" style="background:{color}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                    <div class="day">{day}</div>
                    <div class="price">${price:.0f}</div>
                    <div class="badge">{diff_text}</div>
                    <div>{occ:.0f}% ⛳</div>
                </div>
                """, unsafe_allow_html=True)

                # ✅ ANALYTICS BUTTON — open modal (lazy load)
                if cols[i].button("📊 Analytics", key=f"a_{day}"):
                    st.session_state["analytics_dialog_date"] = date
                    st.session_state.pop("details_dialog_date", None)

                if cols[i].button("View Details", key=f"d_{day}"):
                    st.session_state["details_dialog_date"] = date
                    st.session_state.pop("analytics_dialog_date", None)

            else:
                cols[i].markdown(f"""
                <div class="tile" style="background:#1f2937;">
                    <div class="day">{day}</div>
                    <div>No Data</div>
                </div>
                """, unsafe_allow_html=True)

if "analytics_dialog_date" in st.session_state:
  show_analytics_dialog(st.session_state["analytics_dialog_date"], selected_course)
elif "details_dialog_date" in st.session_state:
  show_details_dialog(st.session_state["details_dialog_date"], selected_course)

# -----------------------------
# ✅ ANALYTICS SECTION (UPDATED)
# -----------------------------

import streamlit.components.v1 as _components

# CSS injected into the PARENT Streamlit page (scripts stripped, CSS kept)
st.markdown("""
<style>
  /* Make the component iframe invisible — zero height, no border */
  iframe[title="streamlit_component"] {
    display: block !important;
    height: 0px !important;
    min-height: 0px !important;
    border: none !important;
    overflow: hidden !important;
  }
</style>
""", unsafe_allow_html=True)

# Full HTML document rendered inside the component iframe.
# JS uses window.parent.document to inject the FAB + chat window
# directly into the Streamlit parent page DOM.
_components.html("""<!DOCTYPE html>
<html><head><meta charset="utf-8"/></head>
<body>
<script>
(function() {
  var p = window.parent.document;

  // ── Inject CSS into parent ──────────────────────────────────────
  if (p.getElementById('gc-style')) return; // already injected
  var style = p.createElement('style');
  style.id = 'gc-style';
  style.textContent = `
    .main .block-container {
      padding-right: clamp(1rem, 34vw, 25rem) !important;
    }
    #gc-fab {
      position: fixed; bottom: 20px; right: 20px;
      width: 62px; height: 62px; border-radius: 50%;
      background: linear-gradient(135deg,#16a34a,#15803d);
      color:#fff; font-size:28px; border:none; cursor:pointer;
      box-shadow:0 6px 24px rgba(0,0,0,.5);
      display:flex; align-items:center; justify-content:center;
      z-index:999999; transition:transform .2s,box-shadow .2s;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    }
    #gc-fab:hover{transform:scale(1.1);box-shadow:0 10px 30px rgba(0,0,0,.6);}
    #gc-badge {
      position:absolute; top:-3px; right:-3px;
      width:18px; height:18px; border-radius:50%;
      background:#ef4444; color:#fff; font-size:10px; font-weight:700;
      display:none; align-items:center; justify-content:center;
      border:2px solid #fff;
    }
    #gc-win {
      position:fixed; bottom:96px; right:20px;
      width:min(370px, calc(100vw - 24px));
      height:min(520px, calc(100vh - 124px));
      background:#111827; border-radius:20px;
      box-shadow:0 20px 60px rgba(0,0,0,.75);
      display:none; flex-direction:column; overflow:hidden;
      border:1px solid #374151; z-index:999998;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    }
    #gc-win.gc-open{display:flex;animation:gcIn .2s ease;}
    @keyframes gcIn{from{opacity:0;transform:translateY(12px) scale(.97)}to{opacity:1;transform:none}}
    #gc-head {
      background:linear-gradient(135deg,#16a34a,#15803d);
      padding:12px 14px; display:flex; align-items:center;
      justify-content:space-between; flex-shrink:0;
    }
    #gc-head-left{display:flex;align-items:center;gap:9px;}
    #gc-avatar {
      width:36px;height:36px;border-radius:50%;
      background:rgba(255,255,255,.2);
      display:flex;align-items:center;justify-content:center;font-size:18px;
    }
    #gc-title{color:#fff;font-weight:700;font-size:14px;}
    #gc-sub{color:rgba(255,255,255,.75);font-size:11px;margin-top:1px;}
    #gc-cls {
      background:none;border:none;color:rgba(255,255,255,.85);
      font-size:18px;cursor:pointer;padding:2px 7px;border-radius:5px;
      transition:background .15s;line-height:1;
    }
    #gc-cls:hover{background:rgba(255,255,255,.2);}
    #gc-msgs {
      flex:1;overflow-y:auto;padding:12px 11px;
      display:flex;flex-direction:column;gap:9px;scroll-behavior:smooth;
    }
    #gc-msgs::-webkit-scrollbar{width:3px;}
    #gc-msgs::-webkit-scrollbar-thumb{background:#374151;border-radius:3px;}
    .gc-msg {
      max-width:87%;padding:9px 13px;border-radius:14px;
      font-size:13px;line-height:1.55;word-break:break-word;
      animation:gcUp .15s ease;
    }
    @keyframes gcUp{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
    .gc-bot{background:#1f2937;color:#e5e7eb;border-bottom-left-radius:3px;align-self:flex-start;}
    .gc-user{background:linear-gradient(135deg,#16a34a,#15803d);color:#fff;border-bottom-right-radius:3px;align-self:flex-end;text-align:right;}
    .gc-typ{background:#1f2937;color:#6b7280;align-self:flex-start;font-style:italic;font-size:12px;}
    #gc-chips{padding:0 11px 9px;display:flex;flex-wrap:wrap;gap:5px;flex-shrink:0;}
    .gc-chip {
      background:#1f2937;color:#9ca3af;border:1px solid #374151;
      border-radius:18px;padding:4px 11px;font-size:11px;
      cursor:pointer;transition:all .15s;white-space:nowrap;
    }
    .gc-chip:hover{background:#16a34a;color:#fff;border-color:#16a34a;}
    #gc-bar {
      display:flex;align-items:center;gap:7px;padding:9px 11px;
      background:#1f2937;border-top:1px solid #374151;flex-shrink:0;
    }
    #gc-inp {
      flex:1;background:#111827;border:1px solid #374151;border-radius:20px;
      color:#f3f4f6;font-size:13px;padding:8px 14px;outline:none;
      transition:border-color .15s;
    }
    #gc-inp:focus{border-color:#16a34a;}
    #gc-inp::placeholder{color:#6b7280;}
    #gc-send {
      width:36px;height:36px;border-radius:50%;
      background:linear-gradient(135deg,#16a34a,#15803d);
      border:none;color:#fff;font-size:16px;cursor:pointer;
      display:flex;align-items:center;justify-content:center;
      flex-shrink:0;transition:transform .15s,opacity .15s;
    }
    #gc-send:hover{transform:scale(1.1);}
    #gc-send:disabled{opacity:.4;cursor:not-allowed;transform:none;}

    @media (max-width: 1200px) {
      .main .block-container {
        padding-right: 1rem !important;
      }
    }

    @media (max-width: 900px) {
      .main .block-container {
        padding-right: 0.75rem !important;
      }

      #gc-win {
        right: 12px;
        bottom: 88px;
        width: calc(100vw - 24px);
        height: min(520px, calc(100vh - 112px));
      }

      #gc-fab {
        right: 12px;
        bottom: 12px;
        width: 58px;
        height: 58px;
        font-size: 26px;
      }
    }

    @media (max-width: 640px) {
      #gc-win {
        right: 10px;
        bottom: 78px;
        width: calc(100vw - 20px);
        height: calc(100vh - 100px);
        border-radius: 16px;
      }

      #gc-fab {
        right: 10px;
        bottom: 10px;
        width: 54px;
        height: 54px;
        font-size: 24px;
      }
    }
  `;
  p.head.appendChild(style);

  // ── Inject HTML into parent body ────────────────────────────────
  var wrap = p.createElement('div');
  wrap.id = 'gc-root';
  wrap.innerHTML = `
    <button id="gc-fab" title="Golf Analytics AI">
      ⛳<span id="gc-badge">1</span>
    </button>
    <div id="gc-win">
      <div id="gc-head">
        <div id="gc-head-left">
          <div id="gc-avatar">🤖</div>
          <div>
            <div id="gc-title">Golf Analytics AI</div>
            <div id="gc-sub">Prices · Availability · Market Rates</div>
          </div>
        </div>
        <button id="gc-cls" title="Close">✕</button>
      </div>
      <div id="gc-msgs">
        <div class="gc-msg gc-bot">
          👋 Hi! I'm your Golf Analytics Assistant.<br><br>
          Ask me anything about tee time prices, availability, occupancy, or market rates.
        </div>
      </div>
      <div id="gc-chips">
        <button class="gc-chip" data-q="Average price for stonegate golf club?">💰 Stonegate price</button>
        <button class="gc-chip" data-q="Which course has the highest occupancy?">📈 Top occupancy</button>
        <button class="gc-chip" data-q="Available tee times on GolfNow?">🟢 GolfNow slots</button>
        <button class="gc-chip" data-q="Compare prices across all courses">📊 Compare courses</button>
      </div>
      <div id="gc-bar">
        <input id="gc-inp" type="text" placeholder="Ask about golf data…" autocomplete="off"/>
        <button id="gc-send" title="Send">&#10148;</button>
      </div>
    </div>
  `;
  p.body.appendChild(wrap);

  // ── Wire up JS in parent context ────────────────────────────────
  var API    = 'http://localhost:8001/chat';
  var isOpen = false;
  var busy   = false;

  var fab   = p.getElementById('gc-fab');
  var win   = p.getElementById('gc-win');
  var badge = p.getElementById('gc-badge');
  var msgs  = p.getElementById('gc-msgs');
  var chips = p.getElementById('gc-chips');
  var inp   = p.getElementById('gc-inp');
  var send  = p.getElementById('gc-send');
  var cls   = p.getElementById('gc-cls');

  function toggle() {
    isOpen = !isOpen;
    win.classList.toggle('gc-open', isOpen);
    badge.style.display = 'none';
    if (isOpen) setTimeout(function(){ inp.focus(); }, 60);
  }

  function addMsg(html, cn) {
    var d = p.createElement('div');
    d.className = 'gc-msg ' + cn;
    d.innerHTML = html;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    return d;
  }

  function doSend() {
    if (busy) return;
    var q = inp.value.trim();
    if (!q) return;
    inp.value = '';
    addMsg(q, 'gc-user');
    chips.style.display = 'none';
    var typing = addMsg('⛳ Thinking…', 'gc-typ');
    busy = true; send.disabled = true;

    fetch(API, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q, top_k:8})
    })
    .then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
    .then(function(data){
      typing.remove();
      addMsg(data.answer.replace(/\\n/g,'<br>'), 'gc-bot');
    })
    .catch(function(){
      typing.remove();
      addMsg('⚠️ Cannot reach backend. Is FastAPI running on port 8001?<br><code style="font-size:11px">.venv\\Scripts\\python.exe -m uvicorn backend.app:app --port 8001 --reload</code>', 'gc-bot');
    })
    .finally(function(){ busy=false; send.disabled=false; inp.focus(); });
  }

  fab.addEventListener('click', toggle);
  cls.addEventListener('click', toggle);
  send.addEventListener('click', doSend);
  inp.addEventListener('keydown', function(e){ if(e.key==='Enter') doSend(); });
  p.querySelectorAll('.gc-chip').forEach(function(btn){
    btn.addEventListener('click', function(){
      inp.value = btn.getAttribute('data-q');
      doSend();
    });
  });

  setTimeout(function(){ if(!isOpen) badge.style.display='flex'; }, 1500);
})();
</script>
</body></html>
""", height=0, scrolling=False)
