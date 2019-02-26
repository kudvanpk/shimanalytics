import cluster
import shim
import template_table_gen
import argparse

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--logfile',  type=str, required=False, help='file name')
        parser.add_argument('--alchemy_env',  type=str, required=True, help='prod:dev')
        parser.add_argument('--monitors',  type=str, required=True, help='metric file')
        parser.add_argument('--creds',  type=str, required=True, help='creds file')
        parser.add_argument('--alert_location',  type=str, required=False, help='location to send alerts')
        parser.add_argument('--sleep_time',  type=str, required=False, help='sleep between metric pulls')
        args = parser.parse_args()
        print(args)
#        shim_listener = shim.ShimListener(args, logfile)
#        shim_listener.get_logs_alchemy()
#        TTG= template_table_gen.TemplateTableGen(args)
#        print(TTG)
        num_nodes = TTG.log_templates_and_tables(logfile + "_1", "template_dict.json", "/root/op/shim-analytics/src/results5")
        num_nodes=100
        LC = cluster.LibCluster (num_nodes, "/root/op/shim-analytics/src/results5", template_dict="template_dict.json")
        print(LC)
        LC.build_graph_analyze()

    except Exception:
        print('Top Level Exception:')
