import {QdrantClient, QdrantClientParams} from '@qdrant/js-client-rest';
import { Pipeline, pipeline } from '@xenova/transformers';
import {v4 as uuidV4} from 'uuid';

interface ICollectionCreateOptions{
    timeout?:number|undefined, 
    init_from?:string, 
    on_disk_payload?:boolean
}

interface ICollectionSearchOptions{
    filter?:TMongoStyleFilter[],
    limit?:number,
    offset?:number,
    score_threshold?:number
}

type TAnd = {$and:Record<string,any>[]};
type ToOr = {$or:Record<string,any>[]};
type TNot = {$not:Record<string,any>[]};
type TMongoStyleFilter = TAnd|ToOr|TNot|Record<string,any>;

class Qwrap{
    private client: QdrantClient;
    private pipe: Pipeline;

    constructor(params?:QdrantClientParams){
        params = params || {host:'localhost',port:6333};
        this.client = new QdrantClient(params);
        this.pipe = pipeline('feature-extraction') as unknown as Pipeline;
    }

    public async createCollection(collection_name:string, options?:ICollectionCreateOptions){
        let opts = options || {timeout: 1000, vectors: {size:384,distance:'Cosine'}}
        if(!('vectors' in opts)){
            opts = {
                ...opts,
                vectors:{
                    size:384,
                    distance:'Cosine'
                }
            }
        }
        return await this.client.createCollection(collection_name,opts as any);
    }

    public async deleteCollection(collection_name:string,args?:{timeout?:number}){
        return await this.client.deleteCollection(collection_name,args);
    }

    public async getCollection(collection_name:string){
        return await this.client.getCollection(collection_name);
    }

    public async getCollections(){
        return await this.client.getCollections();
    }

    public async insert(collection_name:string, data:{data:string,payload?:Record<string,any>}[], args?:{timeout?:number}){
        const collection_list = (await this.client.getCollections()).collections.map((item)=>item.name);
        if(!collection_list.includes(collection_name)){
            await this.client.createCollection(collection_name,{
                timeout: 1000,
                vectors: {
                    size:384,
                    distance:'Cosine'
                }
            });
        }
        const collection = await this.client.getCollection(collection_name);
        if(collection.status !== "green"){
            throw new Error('Collection is not ready');
        }
        if(this.pipe instanceof Promise){
            this.pipe = await this.pipe;
        }
        let vectors = data.map((item)=>this.pipe(item.data));
        vectors = await Promise.all(vectors);
        const payloads = data.map((item)=>{
            return {_qData:item.data,...item.payload} || {}
        });
        return await this.client.upsert(collection_name,{
            batch: {
                vectors:vectors.map(v=>Array.from(v.data as Float32Array)), 
                ids: data.map((item)=>item.payload?.id || uuidV4()),
                payloads
            }
        });
    }

    public async delete(collection_name:string, filter:Record<string,any>){
        let kvs = Object.entries(filter);
        let filters = kvs.map(([k,v])=>{
            return {
                key:k,
                match:{
                    value:v
                }
            }
        });
        return await this.client.delete(collection_name,{filter:{must:filters}});
    }

    public async search(collection_name:string,query:string, opts?:ICollectionSearchOptions){
        if(this.pipe instanceof Promise){
            this.pipe = await this.pipe;
        }
        const vector = await this.pipe(query);
        const options:Omit<ICollectionSearchOptions,'filter'> & Record<string,any> & {filter?:Record<string,any>} = {
            vector: Array.from(vector.data),
            with_vector:false,
            with_payload:true,
        }
        if(opts){
            if(opts.filter){
                options['filter'] = this.MongoFiltersToQdrantFilters(opts.filter);
            }
            if(opts.limit){
                options['limit'] = opts.limit;
            }
            if(opts.offset){
                options['offset'] = opts.offset;
            }
            if(opts.score_threshold){
                options['score_threshold'] = opts.score_threshold;
            }
        }
        return await this.client.search(collection_name,options as any);
    }

    private MongoFiltersToQdrantFilters(filters:TMongoStyleFilter[]){
        let qdrantFilters:Record<string,any> = {};
        filters.forEach((filter)=>{
            if('$and' in filter){
                qdrantFilters['must'] = this.MongoFiltersToQdrantFilters(filter['$and']);
            }else if('$or' in filter){
                qdrantFilters['should'] = this.MongoFiltersToQdrantFilters(filter['$or']);
            }else if('$not' in filter){
                qdrantFilters['must_not'] = this.MongoFiltersToQdrantFilters(filter['$not']);
            }else{
                let kvs = Object.entries(filter);
                let musts = kvs.map(([k,v])=>{
                    return {
                        key:k,
                        match:{
                            value:v
                        }
                    }
                });
                qdrantFilters['must'] = musts;
            }
        });
        return qdrantFilters;
    }
}

export default Qwrap;