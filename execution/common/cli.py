"""Small standard-library CLI shared by every execution entry point."""
import argparse, copy, importlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def merge(base,changes):
    for key,value in changes.items():
        if key not in base: raise ValueError(f'Unknown configuration option: {key}')
        # Nested maps are complete replacements: a composition can remove categories.
        base[key]=value
    return base

def parse(name,description,argv=None):
    initial=argparse.ArgumentParser(add_help=False)
    initial.add_argument('--config',type=Path)
    first,_=initial.parse_known_args(argv)
    defaults=json.loads((ROOT/'execution/configs'/f'{name}.json').read_text())
    config=copy.deepcopy(defaults)
    if first.config: merge(config,json.loads(first.config.read_text()))
    ap=argparse.ArgumentParser(description=description,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--config',type=Path,help='JSON overrides. Paths are relative to the current working directory.')
    ap.add_argument('--dry-run',action='store_true',help='Validate and print the resolved configuration without running the stage.')
    ap.add_argument('--print-config',action='store_true',help='Print resolved JSON without running the stage.')
    for key,value in config.items():
        option='--'+key.replace('_','-');default=defaults[key]
        helptext=f'{key.replace("_"," ")}'+(' (JSON object/list)' if isinstance(default,(dict,list)) or default is None else '')
        if isinstance(default,bool):ap.add_argument(option,action=argparse.BooleanOptionalAction,default=value,help=helptext)
        elif isinstance(default,list) and default and not isinstance(default[0],(dict,list)):
            ap.add_argument(option,nargs='+',type=type(default[0]),default=value,help=key.replace('_',' ')+' (space-separated)')
        elif default is None and key=='resume':ap.add_argument(option,type=str,default=value,help='auto or a checkpoint path')
        elif isinstance(default,(dict,list)) or default is None:ap.add_argument(option,type=json.loads,default=value,help=helptext)
        else:ap.add_argument(option,type=type(default),default=value,help=helptext)
    args=ap.parse_args(argv);resolved={k:getattr(args,k) for k in defaults}
    return ap,args,resolved

def launch(name,package,description):
    ap,args,config=parse(name,description)
    try:
        mod=importlib.import_module(f'execution.components.{package}.workflow')
        mod.validate(config)
        if args.dry_run or args.print_config:
            print(json.dumps(config,indent=2))
            if args.dry_run and hasattr(mod,'command'):print('\nTraining command:\n'+__import__('shlex').join(mod.command(config)))
            return
        mod.run(config)
    except (ValueError,FileNotFoundError,FileExistsError) as exc:ap.exit(2,f'{name}: {exc}\n')
