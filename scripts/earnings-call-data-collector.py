import pandas as pd
import duckdb

from defeatbeta_api.data.ticker import Ticker


DIS = Ticker("DIS")

transcripts = DIS.earning_call_transcripts()

transcripts_list = transcripts.get_transcripts_list()

records = []
for i, row in enumerate(transcripts_list.itertuples(index=False)):
    fiscal_year = row.fiscal_year
    fiscal_quarter = row.fiscal_quarter

    df = transcripts.get_transcript(fiscal_year, fiscal_quarter)  # This is a DataFrame
    df["fiscal_year"] = fiscal_year
    df["fiscal_quarter"] = fiscal_quarter
    records.append(df)


df_all = pd.concat(records, ignore_index=True)

# Save to DuckDB
con = duckdb.connect("DIS_transcripts.duckdb")
con.register("temp_df", df_all)
con.execute("CREATE OR REPLACE TABLE transcripts AS SELECT * FROM temp_df")
con.close()
