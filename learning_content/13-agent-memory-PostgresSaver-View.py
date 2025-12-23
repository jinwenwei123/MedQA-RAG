## 安装 PostgreSQL
# https://www.postgresql.org/download/windows/
# 1）user:postgres/xxxxxx
# 2）port:5432
# 3）把 D:\PostgreSQL\18\bin 配置到环境变量path里面
from anyio.lowlevel import checkpoint
## 工具
# psql -U postgres -h localhost -p 5432

## 主要命令：
#   \l 列出所有数据库
#   \dt 列出所有数据表
#   \d <table>  查看表结构
#   删除表：DROP TABLE checkpoint_blobs, checkpoint_migrations, checkpoint_writes, checkpoints CASCADE;
#   SELECT * from <table> 查看数据
#   \q 退出

## pip install psycopg langgraph-checkpoint-postgres


from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

DB_URI = "postgresql://postgres:123456@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpoints = checkpointer.list({"configurable": {"thread_id": "1"}})

    for checkpoint in checkpoints:
        messages = checkpoint[1]["channel_values"]["messages"]
        for message in messages:
            message.pretty_print()
        break
