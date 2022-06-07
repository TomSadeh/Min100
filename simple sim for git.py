#Importing the relevant libraries.
import pandas as pd
import numpy as np

#Defining two useful functions to calculate income tax and weighted median.
def compute_tax(salary, levels, pcts, zichuy = 2.25, schum_zichuy = 216, max_salary = 0, max_tax = 0):
    """
    A function that calculates the amount of Income Tax or National Security Tax
    a person needs to pay according to the Israeli tax laws.

    Parameters
    ----------
    salary : Float
        The salary from which the tax will be deducted.
    levels : Iterable
        A list of the tax brackets.
    pcts : Iterable
        A list of the tax brackets percents.
    zichuy : Float, optional
        The amount of zichuy points the salary earner has. The default is 2.25.
    schum_zichuy : Float, optional
        The Value of a single zichuy point. The default is 219.
    max_salary : Float, optional
        An optional maximum salary. The default is 0.
    max_tax : Float, optional
        An optional maximum tax. The default is 0.

    Returns
    -------
    Float
        The amount of tax which will be deducted from the salary.
        
    Required libraries
    ---------
    None.    
    """  
    #Returning the max tax if the conditions are met.
    tax = 0 
    if max_tax > 0 and max_salary > 0 and salary >= max_salary:
        return max_tax
    
    #The loop which calculates the tax.    
    for pct, bottom, top in zip(pcts, levels, levels[1:] + [salary]): 
        if salary - bottom <= 0 : 
            break 
        if salary > top:
            tax += pct * (top - bottom)  
        else: 
            tax += pct * (salary - bottom)
   
    #If the tax is less than the tax credit then return zero.     
    if tax <= (zichuy * schum_zichuy):
        return 0
    
    #If not, return the tax minus the tax credit.
    else:
        return tax - (zichuy * schum_zichuy)

def weighted_median(data, weights, interpolate = False):
    """
    A function that calculates the weighted median of a given series of values 
    by using a series of weights.
    
    Parameters
    ----------
    data : Iterable
        The data which the function calculates the median for.
    weights : Iterable
        The weights the function uses to calculate an weighted median.
    interpolate : bool
        A boolean argument for interpolating the median, if necessary.
        The default value is False.
        
    Returns
    -------
    numpy.float64
        The function return the weighted median.
        
    Required libraries
    ---------
    Numpy.
    """
    #Forcing the data to a numpy array.
    data = np.array(data)
    weights = np.array(weights)
    
    #Sorting the data and the weights.
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
   
    #Calculating the cumulative sum of the weights.
    sn = np.cumsum(sorted_weights)
    
    #Calculating the threshold.
    threshold = sorted_weights.sum()/2
   
    #Interpolating the median and returning it.
    if interpolate:
        return np.interp(0.5, (sn - 0.5 * sorted_weights) / np.sum(sorted_weights), sorted_data)
    
    #Returning the first value that equals or larger than the threshold.
    else:
        return sorted_data[sn >= threshold][0]

#Defining dictionaries with tax brackets, and max tax and salary for national insurance tax.
income_tax = dict(levels = [0, 6310, 9050, 14530, 20200, 42030, 54130],
                  pcts = [0.1, 0.14, 0.2, 0.31, 0.35, 0.47, 0.5])          
btl_tax = dict(levels = [0, 6164],
               pcts = [0.035, 0.12],
               max_salary = 43890,
               max_tax = 4743)

#Assigning a variable for the minimum wage calculation.
new_min_wage = 100

#Importing the files. Enter your file address here.
base_address = r''
prat = pd.read_csv(base_address + '\\' + 'H20191021dataprat.csv')
mbs = pd.read_csv(base_address + '\\' + 'H20191021datamb.csv')

#Making all the columns lower-cased.
prat.columns = prat.columns.str.lower()
mbs.columns = mbs.columns.str.lower()

#Setting an index for the households table.
mbs.set_index('misparmb', inplace = True)

#Filtering the individuals dataframe for only individuals that are workers with salary only and have positive work hours.
mask_waged = prat['i111prat'] > 0 
mask_employee = prat['i112prat'] == 0
mask_hours = (prat['sh_shavua'] * prat['shavuot']) > 0
prat = prat[mask_waged & mask_employee & mask_hours ]

#Calculating salary per work hour, by dividing the monthly salary by the number of weeks the worker worked multiplied by the number of hours per week he/she works.
prat['hourly'] = prat['i111prat'] / (prat['sh_shavua'] * prat['shavuot'])
prat['new_hourly'] = prat['hourly']

#Filtering workers below 20 NIS per hour, they are mostly soldiers in the standing army. Then assigning the new minimum wage for all the workers that earn below it.
mask = prat['hourly'].between(20,new_min_wage)
prat.loc[mask, 'new_hourly'] = new_min_wage

#Calculating the new gross wage by multipling the new hourly wage by the work hours.
prat['new_i111prat'] = prat['i111prat']
prat.loc[mask, 'new_i111prat'] = prat.loc[mask, 'new_hourly'] * (prat.loc[mask, 'sh_shavua'] * prat.loc[mask, 'shavuot'])

#Dropping nans in the new column.
prat.dropna(subset = 'new_hourly', inplace = True)

#Calculating the new net income, by adding the old net income to the difference between the new gross income and the old gross income. 
#Then from this, substracting the difference between the old gross income tax (without zichuy points), and new gross income tax. 
#Then substracing the difference between the old national insurance tax and the new national insurance tax. 
prat['new_net'] = prat['net_work_prat']
prat.loc[mask, 'new_net'] = (prat.loc[mask, 'net_work_prat'] + \
                            (prat.loc[mask, 'new_i111prat'] - prat.loc[mask, 'i111prat'])) - \
                            (prat.loc[mask, 'new_i111prat'].apply(compute_tax, args = (income_tax['levels'], income_tax['pcts']), zichuy = 0) - \
                             prat.loc[mask, 'i111prat'].apply(compute_tax, args = (income_tax['levels'], income_tax['pcts']), zichuy = 0)) - \
                            (prat.loc[mask, 'new_i111prat'].apply(compute_tax, args = (btl_tax['levels'], btl_tax['pcts']), zichuy = 0, max_salary = btl_tax['max_salary'], max_tax = btl_tax['max_tax']) - \
                             prat.loc[mask, 'i111prat'].apply(compute_tax, args = (btl_tax['levels'], btl_tax['pcts']), zichuy = 0, max_salary = btl_tax['max_salary'], max_tax = btl_tax['max_tax']))

#Summing the income of the individuals by households.                                
grouped = prat.groupby('misparmb').agg(np.sum)[['net_work_prat', 'new_net']]

#Adding the new income to the households, by adding the difference between the new income from work and the old income from work. 
mbs['new_net'] = mbs['net'] + grouped['new_net'] - grouped['net_work_prat']
mask = mbs['new_net'].isna()
mbs.loc[mask, 'new_net'] = mbs.loc[mask, 'net']

#Calculating income per standard capita by dividing the net income of the households by the number of standard capitas in each household.
mbs['net_to_nefesh'] = mbs['net'] / mbs['nefeshstandartit']
mbs['new_net_to_nefesh'] = mbs['new_net'] / mbs['nefeshstandartit']

#Calculating the poverty lines by dividing the median of the income per capita by 2. We use weighted income according to LAMAS instructions.
poverty_line = dict(new = weighted_median(mbs['new_net_to_nefesh'], mbs['weight']) / 2,
                    old = weighted_median(mbs['net_to_nefesh'], mbs['weight']) / 2)

#Calculating the poverty ratio by summing the weights of all the households that are below the poverty line and dividing by the sum of weights of the entire survey sample.
poverty_ratio = dict(new = mbs.loc[mbs['new_net_to_nefesh'] <= poverty_line['new'], 'weight'].sum() / mbs['weight'].sum(),
                     old = mbs.loc[mbs['net_to_nefesh'] <= poverty_line['old'], 'weight'].sum() / mbs['weight'].sum())

#Printing the results.
print('Old Poverty Ratio: {}\nNew Poverty Ratio: {}'.format(poverty_ratio['old'], poverty_ratio['new']))